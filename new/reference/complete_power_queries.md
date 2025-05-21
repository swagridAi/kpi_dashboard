# Updated Power Query (M) Code for Essential SLO Dashboard

## Overview

This document provides updated Power Query (M) code for implementing a streamlined SLO Dashboard. All external data sources have been consolidated to a single Excel file with multiple sheets.

**Consolidated Data Source:**
- All data now comes from a single Excel file
- Each required dataset is stored in a separate sheet
- File path is defined as a parameter for easy updating

**Supported Core KPIs:**
1. **Lead Time** - Creation to work start (business days)
2. **Cycle Time** - Work start to completion (business days)
3. **Response Time** - End-to-end resolution (business days)
4. **Throughput** - Tickets completed per period
5. **Service Quality** - SLO achievement percentage
6. **Issue Resolution Time** - Average resolution time

---

## File Path Parameter

```m
// Define file path as a parameter
let
    Source = "C:\Data\SLO_Dashboard_Data.xlsx", // Change this to your actual file path
    Parameter = Source
in
    Parameter
```

---

## Core Fact Tables

### 1. Jira_Snapshot (New Query)
```m
let
    // Get data from the consolidated Excel file
    Source = Excel.Workbook(File.Contents(SLO_Excel_FilePath), null, true),
    JiraSnapshotSheet = Source{[Item="JiraSnapshot",Kind="Sheet"]}[Data],
    PromotedHeaders = Table.PromoteHeaders(JiraSnapshotSheet, [PromoteAllScalars=true]),
    
    // Transform data types
    TypedData = Table.TransformColumnTypes(PromotedHeaders, {
        {"key", type text},
        {"issue_type", type text},
        {"epic_name", type text},
        {"status", type text},
        {"created", type datetime},
        {"updated", type datetime},
        {"resolution_date", type datetime},
        {"assignee_display_name", type text},
        {"summary", type text},
        {"active", type logical}
    })
in
    TypedData
```

### 2. Fact_Ticket_Summary
```m
let
    // ===== DIRECT DATA SOURCES =====
    // Load all required data directly from Excel sheets
    ExcelSource = Excel.Workbook(File.Contents(SLO_Excel_FilePath), null, true),
    
    // 1. Load Jira data directly
    JiraSnapshotSheet = ExcelSource{[Item="JiraSnapshot",Kind="Sheet"]}[Data],
    JiraData = Table.PromoteHeaders(JiraSnapshotSheet, [PromoteAllScalars=true]),
    
    // 2. Load capability mapping directly
    CapabilityMappingSheet = ExcelSource{[Item="CapabilityMapping",Kind="Sheet"]}[Data],
    CapabilityMappingData = Table.PromoteHeaders(CapabilityMappingSheet, [PromoteAllScalars=true]),
    
    // 3. Load capability dimension directly
    CapabilitySheet = ExcelSource{[Item="SLOTargets",Kind="Sheet"]}[Data],
    CapabilityData = Table.PromoteHeaders(CapabilitySheet, [PromoteAllScalars=true]),
    
    // 4. Load default SLA directly
    DefaultSLASheet = ExcelSource{[Item="DefaultSLA",Kind="Sheet"]}[Data],
    DefaultSLAData = Table.PromoteHeaders(DefaultSLASheet, [PromoteAllScalars=true]),
    
    // ===== JIRA DATA TRANSFORMATIONS =====
    JiraTyped = Table.TransformColumnTypes(JiraData, {
        {"key", type text},
        {"issue_type", type text},
        {"status", type text},
        {"created", type datetime},
        {"updated", type datetime},
        {"resolution_date", type datetime},
        {"assignee_display_name", type text},
        {"summary", type text},
        {"active", type logical}
    }),
    
    // Filter active tickets
    FilterActive = Table.SelectRows(JiraTyped, each 
        [active] = true and 
        [key] <> null and 
        [created] <> null
    ),
    
    // Add project code
    AddProject = Table.AddColumn(FilterActive, "project", each 
        try Text.Start([key], Text.PositionOf([key], "-")) otherwise null),
    
    // Normalize columns for matching
    AddNormalizedFields = Table.AddColumn(AddProject, "normalized_issue_type", each
        try Text.Lower(Text.Trim([issue_type])) otherwise null),
    
    AddNormalizedProject = Table.AddColumn(AddNormalizedFields, "normalized_project", each
        try Text.Lower(Text.Trim([project])) otherwise null),
    
    AddNormalizedEpicName = if Table.HasColumns(AddNormalizedProject, "epic_name") then
        Table.AddColumn(AddNormalizedProject, "normalized_epic_name", each
            try Text.Lower(Text.Trim([epic_name])) otherwise null)
    else
        AddNormalizedProject,
    
    // Add date columns
    AddDateColumns = Table.AddColumn(AddNormalizedEpicName, "CreatedDate", each Date.From([created])),
    AddResolvedDate = Table.AddColumn(AddDateColumns, "ResolvedDate", each 
        if [resolution_date] <> null then Date.From([resolution_date]) else null),
    AddUpdatedDate = Table.AddColumn(AddResolvedDate, "UpdatedDate", each Date.From([updated])),
    
    // Calculate resolution time
    AddResolutionTimeDays = Table.AddColumn(AddUpdatedDate, "ResolutionTimeDays", each
        if [resolution_date] <> null then
            Duration.Days([resolution_date] - [created])
        else
            Duration.Days(DateTime.LocalNow() - [created])
    ),
    
    // Add status flags
    AddIsResolved = Table.AddColumn(AddResolutionTimeDays, "IsResolved", each [resolution_date] <> null),
    
    AddIsCompleted = Table.AddColumn(AddIsResolved, "Is_Completed", each
        let
            CompletedStatuses = {"Done", "Closed", "Resolved", "Fixed", "Completed"}
        in
            List.Contains(CompletedStatuses, [status])
    ),
    
    // ===== PREPARE CAPABILITY MAPPING =====
    CapabilityMappingTyped = Table.TransformColumnTypes(CapabilityMappingData, {
        {"CapabilityKey", type text},
        {"IssueType", type text},
        {"IsActive", type logical}
    }),
    
    ActiveMappings = Table.SelectRows(CapabilityMappingTyped, each [IsActive] = true),
    
    // ===== JOIN WITH CAPABILITY MAPPING =====
    JoinCapabilityMapping = Table.NestedJoin(AddIsCompleted, {"issue_type"}, 
        ActiveMappings, {"IssueType"}, "CapabilityMapping", JoinKind.LeftOuter),
    
    ExpandCapabilityMapping = Table.ExpandTableColumn(JoinCapabilityMapping, "CapabilityMapping", 
        {"CapabilityKey"}, {"MappedCapabilityKey"}),
    
    AddFinalCapabilityKey = Table.AddColumn(ExpandCapabilityMapping, "FinalCapabilityKey", each 
        [MappedCapabilityKey]),
    
    // ===== PREPARE CAPABILITY DIMENSION =====
    CapabilityTyped = Table.TransformColumnTypes(CapabilityData, {
        {"CapabilityKey", type text},
        {"ResponseTimeTargetDays", type number},
        {"IsActive", type logical}
    }),
    
    ActiveCapabilities = Table.SelectRows(CapabilityTyped, each [IsActive] = true),
    
    // ===== PREPARE DEFAULT SLA =====
    DefaultSLATyped = Table.TransformColumnTypes(DefaultSLAData, {
        {"TicketType", type text},
        {"SLA_Days", type number},
        {"IsActive", type logical}
    }),
    
    ActiveSLAs = Table.SelectRows(DefaultSLATyped, each [IsActive] = true),
    
    // ===== SLA TARGET CALCULATION =====
    // Join with capability dimension
    JoinCapability = Table.NestedJoin(AddFinalCapabilityKey, {"FinalCapabilityKey"}, 
        ActiveCapabilities, {"CapabilityKey"}, "CapabilityData", JoinKind.LeftOuter),
    
    ExpandCapability = Table.ExpandTableColumn(JoinCapability, "CapabilityData", 
        {"ResponseTimeTargetDays"}, {"CapabilityResponseTimeTarget"}),
    
    // Join with default SLA
    JoinDefaultSLA = Table.NestedJoin(ExpandCapability, {"issue_type"}, 
        ActiveSLAs, {"TicketType"}, "DefaultSLA", JoinKind.LeftOuter),
    
    ExpandDefaultSLA = Table.ExpandTableColumn(JoinDefaultSLA, "DefaultSLA", 
        {"SLA_Days"}, {"DefaultSLADays"}),
    
    // Calculate SLA target using hierarchy
    AddSLATarget = Table.AddColumn(ExpandDefaultSLA, "ResponseTimeTargetDays", each
        if [CapabilityResponseTimeTarget] <> null then [CapabilityResponseTimeTarget]
        else if [DefaultSLADays] <> null then [DefaultSLADays]
        else 5
    ),
    
    // Calculate SLA achievement
    AddMetSLA = Table.AddColumn(AddSLATarget, "Met_SLA", each
        if [IsResolved] = true and [ResponseTimeTargetDays] <> null then
            [ResolutionTimeDays] <= [ResponseTimeTargetDays]
        else null
    ),
    
    AddIsOverdue = Table.AddColumn(AddMetSLA, "IsOverdue", each
        if [IsResolved] = false and [ResponseTimeTargetDays] <> null then
            [ResolutionTimeDays] > [ResponseTimeTargetDays]
        else false
    ),
    
    // Add status timing
    AddDaysInCurrentStatus = Table.AddColumn(AddIsOverdue, "DaysInCurrentStatus", each
        Duration.Days(DateTime.LocalNow() - [updated])
    ),
    
    // Validate data
    ValidateData = Table.SelectRows(AddDaysInCurrentStatus, each 
        [ResolutionTimeDays] >= 0 and
        ([IsResolved] = false or [resolution_date] <> null)
    ),
    
    // Remove helper columns
    RemoveHelpers = Table.RemoveColumns(ValidateData, {
        "normalized_issue_type", "normalized_project", "normalized_epic_name",
        "CapabilityData", "DefaultSLA", "CapabilityResponseTimeTarget", "DefaultSLADays",
        "MappedCapabilityKey"
    }),
    
    // Final type conversion
    FinalTypes = Table.TransformColumnTypes(RemoveHelpers, {
        {"ResolutionTimeDays", Int64.Type},
        {"DaysInCurrentStatus", Int64.Type},
        {"ResponseTimeTargetDays", type number},
        {"Met_SLA", type logical},
        {"IsOverdue", type logical}
    })
in
    FinalTypes
```

### 3. Fact_Status_Change
```m
let
    // ===== DATA SOURCE FROM CONSOLIDATED EXCEL =====
    Source = Excel.Workbook(File.Contents(SLO_Excel_FilePath), null, true),
    StatusChangeSheet = Source{[Item="StatusChangeData",Kind="Sheet"]}[Data],
    PromotedHeaders = Table.PromoteHeaders(StatusChangeSheet, [PromoteAllScalars=true]),
    
    // ===== INITIAL FILTERING =====
    FilterActive = Table.SelectRows(PromotedHeaders, each 
        [active] = true and 
        [key] <> null and 
        [change_created] <> null
    ),
    
    // ===== DATA TYPES =====
    TypedData = Table.TransformColumnTypes(FilterActive, {
        {"id", Int64.Type},
        {"key", type text},
        {"change_created", type datetime},
        {"from_string", type text},
        {"to_string", type text},
        {"active", type logical}
    }),
    
    // ===== SORT FOR DURATION CALCULATIONS =====
    SortedData = Table.Sort(TypedData, {{"key", Order.Ascending}, {"change_created", Order.Ascending}}),
    
    // ===== ADD INDEX FOR WINDOW FUNCTIONS =====
    AddIndex = Table.AddIndexColumn(SortedData, "RowIndex", 0),
    
    // ===== CALCULATE PREVIOUS CHANGE TIME =====
    // Step 1: Sort and index (global index not used after grouping)
    Sorted = Table.Sort(TypedData, {{"key", Order.Ascending}, {"change_created", Order.Ascending}}),

    // Step 2: Group by 'key' and compute PreviousChangeTime within each group
    Grouped = Table.Group(Sorted, {"key"}, {
        {"AllRows", each 
            let
                T = Table.AddIndexColumn(_, "LocalIndex", 0),
                AddPrev = Table.AddColumn(T, "PreviousChangeTime", each 
                    try T{[LocalIndex]-1}[change_created] 
                    otherwise [change_created])
            in
                Table.RemoveColumns(AddPrev, {"LocalIndex"})
        }
    }),

    // Step 3: Combine grouped tables back into one
    WithPreviousChange = Table.Combine(Grouped[AllRows]),
    
    // ===== DURATION CALCULATIONS =====
    AddDurationCalendar = Table.AddColumn(WithPreviousChange, "DurationCalendarHours", each
        Duration.TotalHours([change_created] - [PreviousChangeTime])
    ),
    
    // ===== BUSINESS HOURS CALCULATION =====
    AddDurationBusiness = Table.AddColumn(AddDurationCalendar, "DurationBusinessHours", each
        let
            StartTime = [PreviousChangeTime],
            EndTime = [change_created],
            StartDate = Date.From(StartTime),
            EndDate = Date.From(EndTime),
            
            // Generate list of dates between start and end
            DateList = List.Dates(StartDate, Duration.Days(EndDate - StartDate) + 1, #duration(1,0,0,0)),
            
            // Filter to business days only (Monday-Friday)
            BusinessDays = List.Select(DateList, each Date.DayOfWeek(_, Day.Monday) < 5),
            
            // Calculate business hours for each day
            BusinessHours = List.Accumulate(BusinessDays, 0, (total, current) =>
                if current = StartDate and current = EndDate then
                    // Same day calculation with business hours check (9 AM - 5 PM)
                    let
                        StartHour = Number.Max(Time.Hour(DateTime.Time(StartTime)), 9),
                        EndHour = Number.Min(Time.Hour(DateTime.Time(EndTime)), 17),
                        Hours = Number.Max(EndHour - StartHour, 0)
                    in total + Hours
                else if current = StartDate then
                    // First day - from start time to 5 PM
                    let
                        StartHour = Number.Max(Time.Hour(DateTime.Time(StartTime)), 9),
                        Hours = Number.Max(17 - StartHour, 0)
                    in total + Hours
                else if current = EndDate then
                    // Last day - from 9 AM to end time
                    let
                        EndHour = Number.Min(Time.Hour(DateTime.Time(EndTime)), 17),
                        Hours = Number.Max(EndHour - 9, 0)
                    in total + Hours
                else
                    // Full business day (8 hours)
                    total + 8
            )
        in Number.Max(BusinessHours, 0)
    ),
    
    // ===== SLO TIMING FLAGS =====
    AddLeadTimeFlag = Table.AddColumn(AddDurationBusiness, "IsLeadTimeStart", each
        let
            LeadTimeStatuses = {"In Progress", "Development", "Analysis", "Working"},
            BacklogStatuses = {"Backlog", "New", "Open", "To Do"},
            FromBacklog = List.Contains(BacklogStatuses, [from_string]),
            ToActive = List.Contains(LeadTimeStatuses, [to_string])
        in FromBacklog and ToActive
    ),
    
    AddCycleTimeFlag = Table.AddColumn(AddLeadTimeFlag, "IsCycleTimeStart", each
        let
            CycleTimeStatuses = {"In Progress", "Development", "Testing", "Review"},
            ToActive = List.Contains(CycleTimeStatuses, [to_string]),
            FromInactive = not List.Contains(CycleTimeStatuses, [from_string])
        in ToActive and FromInactive
    ),
    
    AddResponseTimeFlag = Table.AddColumn(AddCycleTimeFlag, "IsResponseTimeEnd", each
        let
            DoneStatuses = {"Done", "Resolved", "Closed", "Completed"}
        in List.Contains(DoneStatuses, [to_string])
    ),
    
    // ===== ADD RELATIONSHIP KEYS =====
    AddChangeDate = Table.AddColumn(AddResponseTimeFlag, "ChangeDate", each Date.From([change_created])),
    
    
    // ===== FINAL DATA TYPES =====
    FinalTypes = Table.TransformColumnTypes(RemoveHelpers, {
        {"DurationCalendarHours", type number},
        {"DurationBusinessHours", type number},
        {"IsLeadTimeStart", type logical},
        {"IsCycleTimeStart", type logical},
        {"IsResponseTimeEnd", type logical},
        {"ChangeDate", type date}
    })
in
    FinalTypes
```

---

## Essential Dimensions

### 4. Dim_Date (Unchanged)
```m
let
    // ===== GENERATE DATE RANGE =====
    StartDate = #date(2023, 1, 1),
    EndDate = #date(2027, 12, 31),
    DateList = List.Dates(StartDate, Duration.Days(EndDate - StartDate) + 1, #duration(1,0,0,0)),
    
    // ===== CREATE BASE TABLE =====
    DateTable = Table.FromList(DateList, Splitter.SplitByNothing(), {"Date"}),
    
    // ===== CALENDAR HIERARCHY =====
    AddYear = Table.AddColumn(DateTable, "Year", each Date.Year([Date])),
    AddMonth = Table.AddColumn(AddYear, "Month", each Date.Month([Date])),
    AddQuarter = Table.AddColumn(AddMonth, "Quarter", each Date.QuarterOfYear([Date])),
    AddMonthName = Table.AddColumn(AddQuarter, "MonthName", each Date.MonthName([Date])),
    AddQuarterName = Table.AddColumn(AddMonthName, "QuarterName", each "Q" & Text.From([Quarter])),
    
    // ===== WEEK INFORMATION =====
    AddWeekOfYear = Table.AddColumn(AddQuarterName, "WeekOfYear", each Date.WeekOfYear([Date])),
    AddWeekStart = Table.AddColumn(AddWeekOfYear, "WeekStart", each Date.StartOfWeek([Date], Day.Monday)),
    
    // ===== DAY INFORMATION =====
    AddDayOfWeek = Table.AddColumn(AddWeekStart, "DayOfWeek", each Date.DayOfWeek([Date], Day.Monday) + 1),
    AddDayName = Table.AddColumn(AddDayOfWeek, "DayName", each Date.DayOfWeekName([Date])),
    AddIsBusinessDay = Table.AddColumn(AddDayName, "IsBusinessDay", each [DayOfWeek] <= 5),
    AddIsWeekend = Table.AddColumn(AddIsBusinessDay, "IsWeekend", each [DayOfWeek] > 5),
    
    // ===== MONTH BOUNDARIES =====
    AddMonthStart = Table.AddColumn(AddIsWeekend, "MonthStart", each Date.StartOfMonth([Date])),
    AddMonthEnd = Table.AddColumn(AddMonthStart, "MonthEnd", each Date.EndOfMonth([Date])),
    
    // ===== QUARTER BOUNDARIES =====
    AddQuarterStart = Table.AddColumn(AddMonthEnd, "QuarterStart", each Date.StartOfQuarter([Date])),
    AddQuarterEnd = Table.AddColumn(AddQuarterStart, "QuarterEnd", each Date.EndOfQuarter([Date])),
    
    // ===== YEAR BOUNDARIES =====
    AddYearStart = Table.AddColumn(AddQuarterEnd, "YearStart", each Date.StartOfYear([Date])),
    AddYearEnd = Table.AddColumn(AddYearStart, "YearEnd", each Date.EndOfYear([Date])),
    
    // ===== RELATIVE DATE FLAGS =====
    AddIsToday = Table.AddColumn(AddYearEnd, "IsToday", each [Date] = Date.From(DateTime.LocalNow())),
    AddIsYTD = Table.AddColumn(AddIsToday, "IsYTD", each 
        [Year] = Date.Year(DateTime.LocalNow()) and [Date] <= Date.From(DateTime.LocalNow())),
    AddIsQTD = Table.AddColumn(AddIsYTD, "IsQTD", each 
        [Quarter] = Date.QuarterOfYear(DateTime.LocalNow()) and 
        [Year] = Date.Year(DateTime.LocalNow()) and 
        [Date] <= Date.From(DateTime.LocalNow())),
    
    // ===== FISCAL YEAR (Assuming April start) =====
    AddFiscalYear = Table.AddColumn(AddIsQTD, "FiscalYear", each 
        if [Month] >= 4 then [Year] else [Year] - 1),
    AddFiscalQuarter = Table.AddColumn(AddFiscalYear, "FiscalQuarter", each 
        if [Month] >= 4 then [Quarter] + 1 else [Quarter]),
    
    // ===== HOLIDAYS (Basic AUS Holidays) =====
    AddIsHoliday = Table.AddColumn(AddFiscalQuarter, "IsHoliday", each
        let
            Month = [Month],
            Day = Date.Day([Date]),
            Year = [Year]
        in
            // New Year's Day
            (Month = 1 and Day = 1) or
            // Christmas Day
            (Month = 12 and Day = 25) or
            false // Placeholder for additional holidays
    ),
    
    // ===== ADJUST BUSINESS DAY FOR HOLIDAYS =====
    AdjustBusinessDay = Table.ReplaceValue(AddIsHoliday, 
        each [IsBusinessDay], 
        each [IsBusinessDay] and not [IsHoliday], 
        Replacer.ReplaceValue, 
        {"IsBusinessDay"}),
    
    // ===== FINAL DATA TYPES =====
    TypedTable = Table.TransformColumnTypes(AdjustBusinessDay, {
        {"Date", type date},
        {"Year", Int64.Type},
        {"Month", Int64.Type},
        {"Quarter", Int64.Type},
        {"MonthName", type text},
        {"QuarterName", type text},
        {"WeekOfYear", Int64.Type},
        {"WeekStart", type date},
        {"DayOfWeek", Int64.Type},
        {"DayName", type text},
        {"IsBusinessDay", type logical},
        {"IsWeekend", type logical},
        {"MonthStart", type date},
        {"MonthEnd", type date},
        {"QuarterStart", type date},
        {"QuarterEnd", type date},
        {"YearStart", type date},
        {"YearEnd", type date},
        {"IsToday", type logical},
        {"IsYTD", type logical},
        {"IsQTD", type logical},
        {"FiscalYear", Int64.Type},
        {"FiscalQuarter", Int64.Type},
        {"IsHoliday", type logical}
    })
in
    TypedTable
```

### 5. Dim_Capability
```m
let
    // ===== DATA SOURCE FROM CONSOLIDATED EXCEL =====
    Source = Excel.Workbook(File.Contents(SLO_Excel_FilePath), null, true),
    ConfigSheet = Source{[Item="SLOTargets",Kind="Sheet"]}[Data],
    PromotedHeaders = Table.PromoteHeaders(ConfigSheet, [PromoteAllScalars=true]),
    
    // ===== FILTER ACTIVE RECORDS =====
    FilterActive = Table.SelectRows(PromotedHeaders, each [IsActive] = true),
    
    // ===== DATA TYPES =====
    TypedConfig = Table.TransformColumnTypes(FilterActive, {
        {"CapabilityKey", type text},
        {"CapabilityName", type text},
        {"LeadTimeTargetDays", type number},
        {"CycleTimeTargetDays", type number},
        {"ResponseTimeTargetDays", type number},
        {"IsActive", type logical}
    }),
    
    // ===== ADD CALCULATED COLUMNS =====
    AddCapabilityOwner = Table.AddColumn(TypedConfig, "CapabilityOwner", each
        if      [CapabilityKey]="DQ" then "Data Quality Team Lead"
        else if [CapabilityKey]="DE" then "Data Engineering Manager"
        else if [CapabilityKey]="CC" then "Change Control Board"
        else if [CapabilityKey]="RD" then "Data Architecture Team"
        else if [CapabilityKey]="RM" then "Information Governance"
        else "TBD"
    ),
    
    AddBusinessDomain = Table.AddColumn(AddCapabilityOwner, "BusinessDomain", each
        if      [CapabilityKey]="DQ" then "Data Management"
        else if [CapabilityKey]="DE" then "Data Engineering"
        else if [CapabilityKey]="CC" then "IT Operations"
        else if [CapabilityKey]="RD" then "Data Architecture"
        else if [CapabilityKey]="RM" then "Compliance"
        else "General"
    ),

    AddMaturityLevel = Table.AddColumn(AddBusinessDomain, "MaturityLevel", each
        if      [CapabilityKey]="DQ" then "Advanced"
        else if [CapabilityKey]="DE" then "Intermediate"
        else if [CapabilityKey]="CC" then "Advanced"
        else if [CapabilityKey]="RD" then "Intermediate"
        else if [CapabilityKey]="RM" then "Basic"
        else "Basic"
    ),
    
    // ===== ADD METADATA =====
    AddCreatedDate = Table.AddColumn(AddMaturityLevel, "CreatedDate", each Date.From(DateTime.LocalNow())),
    AddLastModified = Table.AddColumn(AddCreatedDate, "LastModified", each Date.From(DateTime.LocalNow())),
    
    // ===== VALIDATION =====
    ValidateTargets = Table.SelectRows(AddLastModified, each 
        [LeadTimeTargetDays] > 0 and 
        [CycleTimeTargetDays] > 0 and 
        [ResponseTimeTargetDays] >= [LeadTimeTargetDays] + [CycleTimeTargetDays]
    )
in
    ValidateTargets
```

### 6. Dim_Status (Unchanged)
```m
let
    // ===== STATIC STATUS DEFINITIONS =====
    Source = #table(
        {"status", "StatusCategory", "StatusOrder", "IncludeInLeadTime", "IncludeInCycleTime", "IncludeInResponseTime"},
        {
            {"Backlog", "To Do", 1, true, false, false},
            {"New", "To Do", 2, true, false, false},
            {"Open", "To Do", 3, true, false, false},
            {"To Do", "To Do", 4, true, false, false},
            {"In Progress", "In Progress", 5, false, true, false},
            {"Development", "In Progress", 6, false, true, false},
            {"Analysis", "In Progress", 7, false, true, false},
            {"Testing", "In Progress", 8, false, true, false},
            {"Review", "In Progress", 9, false, true, false},
            {"Waiting", "Waiting", 10, false, false, false},
            {"Blocked", "Waiting", 11, false, false, false},
            {"On Hold", "Waiting", 12, false, false, false},
            {"Done", "Done", 13, false, false, true},
            {"Resolved", "Done", 14, false, false, true},
            {"Closed", "Done", 15, false, false, true},
            {"Completed", "Done", 16, false, false, true},
            {"Fixed", "Done", 17, false, false, true},
            {"Cancelled", "Done", 18, false, false, true},
            {"Rejected", "Done", 19, false, false, true}
        }
    ),
    
    // ===== ADD BUSINESS LOGIC =====
    AddTimeType = Table.AddColumn(Source, "TimeType", each
        if [IncludeInLeadTime] then "Lead"
        else if [IncludeInCycleTime] then "Cycle"
        else if [IncludeInResponseTime] then "Response"
        else "Other"
    ),
    
    AddIsActiveStatus = Table.AddColumn(AddTimeType, "IsActiveStatus", each
        [StatusCategory] = "In Progress"
    ),
    
    AddIsFinalStatus = Table.AddColumn(AddIsActiveStatus, "IsFinalStatus", each
        [StatusCategory] = "Done"
    ),
    
    AddIsWaitingStatus = Table.AddColumn(AddIsFinalStatus, "IsWaitingStatus", each
        [StatusCategory] = "Waiting"
    ),
    
    // ===== ADD METADATA =====
    AddCreatedDate = Table.AddColumn(AddIsWaitingStatus, "CreatedDate", each Date.From(DateTime.LocalNow())),
    AddIsActive = Table.AddColumn(AddCreatedDate, "IsActive", each true),
    
    // ===== DATA TYPES =====
    TypedStatuses = Table.TransformColumnTypes(AddIsActive, {
        {"status", type text},
        {"StatusCategory", type text},
        {"StatusOrder", Int64.Type},
        {"IncludeInLeadTime", type logical},
        {"IncludeInCycleTime", type logical},
        {"IncludeInResponseTime", type logical},
        {"TimeType", type text},
        {"IsActiveStatus", type logical},
        {"IsFinalStatus", type logical},
        {"IsWaitingStatus", type logical},
        {"CreatedDate", type date},
        {"IsActive", type logical}
    })
in
    TypedStatuses
```

---

## Configuration Tables

### 7. Config_Issue_Type_Mapping
```m
let
    // ===== DATA SOURCE FROM CONSOLIDATED EXCEL =====
    Source = Excel.Workbook(File.Contents(SLO_Excel_FilePath), null, true),
    MappingSheet = Source{[Item="IssueTypeMapping",Kind="Sheet"]}[Data],
    PromotedHeaders = Table.PromoteHeaders(MappingSheet, [PromoteAllScalars=true]),
    
    // ===== FILTER ACTIVE MAPPINGS =====
    FilterActive = Table.SelectRows(PromotedHeaders, each [IsActive] = true),
    
    // ===== DATA TYPES =====
    TypedMapping = Table.TransformColumnTypes(FilterActive, {
        {"issue_type", type text},
        {"CapabilityKey", type text},
        {"Notes", type text},
        {"IsActive", type logical}
    }),
    
    // ===== ADD VALIDATION =====
    AddMappingKey = Table.AddIndexColumn(TypedMapping, "MappingKey", 1, 1),
    AddEffectiveDate = Table.AddColumn(AddMappingKey, "EffectiveDate", each Date.From(DateTime.LocalNow())),
    AddCreatedBy = Table.AddColumn(AddEffectiveDate, "CreatedBy", each "System Administrator"),
    
    // ===== VALIDATION RULES =====
    ValidateMappings = Table.SelectRows(AddCreatedBy, each 
        [issue_type] <> null and 
        [CapabilityKey] <> null and
        Text.Length([issue_type]) > 0 and
        Text.Length([CapabilityKey]) > 0
    ),
    
    // ===== FINAL DATA TYPES =====
    FinalTypes = Table.TransformColumnTypes(ValidateMappings, {
        {"MappingKey", Int64.Type},
        {"EffectiveDate", type date},
        {"CreatedBy", type text}
    })
in
    FinalTypes
```

### 8. Default_SLA_Table
```m
let
    // ===== DATA SOURCE FROM CONSOLIDATED EXCEL =====
    Source = Excel.Workbook(File.Contents(SLO_Excel_FilePath), null, true),
    SLASheet = Source{[Item="DefaultSLA",Kind="Sheet"]}[Data],
    PromotedHeaders = Table.PromoteHeaders(SLASheet, [PromoteAllScalars=true]),
    
    // Alternative: Keep the static table if the sheet isn't available
    CheckForData = if Table.IsEmpty(PromotedHeaders) then
        #table(
            {"TicketType", "SLA_Days", "DefaultCriticality", "ExcludeWeekends", "BusinessDaysOnly", "Notes"},
            {
                {"Bug", 3, "High", true, true, "Critical defects require faster response"},
                {"Task", 5, "Standard", true, true, "Standard business day response target"},
                {"Epic", 10, "Medium", true, true, "Large initiatives allow longer response time"},
                {"Story", 8, "Standard", true, true, "User stories standard processing"},
                {"Sub-task", 2, "Standard", true, true, "Sub-tasks inherit parent priority but faster turnaround"},
                {"Improvement", 7, "Medium", true, true, "Enhancement requests standard timeline"},
                {"New Feature", 15, "Low", true, true, "New features require extended analysis time"},
                {"Change Request", 10, "Medium", true, true, "Change management standard process"},
                {"Incident", 1, "Critical", false, false, "Production incidents have highest priority"},
                {"Service Request", 5, "Standard", true, true, "Standard service delivery"},
                {"Data Quality Task", 3, "High", true, true, "Data quality work requires attention"},
                {"Extract Request", 5, "Standard", true, true, "Data extraction standard timeline"},
                {"Emergency Change", 0.5, "Critical", false, false, "Emergency changes require immediate attention"},
                {"Reference Data Update", 4, "Standard", true, true, "Reference data updates standard process"},
                {"Records Request", 3, "Standard", true, true, "Records management requests"}
            }
        )
    else
        PromotedHeaders,
    
    // ===== ADD METADATA =====
    AddSLAKey = Table.AddIndexColumn(CheckForData, "SLA_Key", 1, 1),
    AddSLAType = Table.AddColumn(AddSLAKey, "SLA_Type", each "Response Time"),
    AddIsActive = Table.AddColumn(AddSLAType, "IsActive", each true),
    AddCreatedDate = Table.AddColumn(AddIsActive, "CreatedDate", each Date.From(DateTime.LocalNow())),
    AddCreatedBy = Table.AddColumn(AddCreatedDate, "CreatedBy", each "System Administrator"),
    AddLastModified = Table.AddColumn(AddCreatedBy, "LastModified", each Date.From(DateTime.LocalNow())),
    
    // ===== ADD BUSINESS RULES =====
    AddHolidayAdjustment = Table.AddColumn(AddLastModified, "HolidayAdjustment", each
        if [ExcludeWeekends] then "Standard" else "None"
    ),
    
    AddEscalationThreshold = Table.AddColumn(AddHolidayAdjustment, "EscalationThreshold", each
        [SLA_Days] * 0.8  // Alert at 80% of SLA
    ),
    
    // ===== VALIDATION =====
    ValidateSLAs = Table.SelectRows(AddEscalationThreshold, each 
        [SLA_Days] > 0 and [SLA_Days] <= 30  // Reasonable SLA range
    ),
    
    // ===== FINAL DATA TYPES =====
    TypedTable = Table.TransformColumnTypes(ValidateSLAs, {
        {"SLA_Key", Int64.Type},
        {"TicketType", type text},
        {"SLA_Days", type number},
        {"DefaultCriticality", type text},
        {"ExcludeWeekends", type logical},
        {"BusinessDaysOnly", type logical},
        {"Notes", type text},
        {"SLA_Type", type text},
        {"IsActive", type logical},
        {"CreatedDate", type date},
        {"CreatedBy", type text},
        {"LastModified", type date},
        {"HolidayAdjustment", type text},
        {"EscalationThreshold", type number}
    })
in
    TypedTable
```

---

## Helper Functions

### 9. Business Hours Calculation Function (Unchanged)
```m
let
    BusinessHoursFunction = (StartTime as datetime, EndTime as datetime) =>
        let
            StartDate = Date.From(StartTime),
            EndDate = Date.From(EndTime),
            
            // Generate list of all dates in range
            DateList = List.Dates(
                StartDate, 
                Duration.Days(EndDate - StartDate) + 1, 
                #duration(1,0,0,0)
            ),
            
            // Filter to business days only (Monday-Friday)
            BusinessDays = List.Select(DateList, each 
                Date.DayOfWeek(_, Day.Monday) < 5
            ),
            
            // Calculate business hours for each day
            BusinessHours = List.Accumulate(BusinessDays, 0, (total, current) =>
                if current = StartDate and current = EndDate then
                    // Same day calculation (consider business hours 9-17)
                    let
                        ActualStart = Number.Max(Time.Hour(DateTime.Time(StartTime)), 9),
                        ActualEnd = Number.Min(Time.Hour(DateTime.Time(EndTime)), 17)
                    in total + Number.Max(ActualEnd - ActualStart, 0)
                else if current = StartDate then
                    // First day - from start time to end of business day
                    let
                        ActualStart = Number.Max(Time.Hour(DateTime.Time(StartTime)), 9)
                    in total + Number.Max(17 - ActualStart, 0)
                else if current = EndDate then
                    // Last day - from start of business day to end time  
                    let
                        ActualEnd = Number.Min(Time.Hour(DateTime.Time(EndTime)), 17)
                    in total + Number.Max(ActualEnd - 9, 0)
                else
                    // Full business day (8 hours)
                    total + 8
            )
        in
            Number.Max(BusinessHours, 0)
in
    BusinessHoursFunction
```

---

## Implementation Notes

### Excel Sheet Structure Required
The consolidated Excel file should have the following sheets:
1. **JiraSnapshot** - Contains the Jira ticket data
2. **StatusChangeData** - Contains status transition history
3. **SLOTargets** - Contains capability definitions and SLA targets
4. **IssueTypeMapping** - Maps issue types to capabilities
5. **CapabilityMapping** - Maps capabilities to ticket types and other attributes
6. **DefaultSLA** - Contains default SLA targets by ticket type

### Key Simplifications Made:
1. **Single Data Source**: All external data comes from a single Excel file
2. **File Path Parameter**: The Excel file path is defined as a parameter for easy updates
3. **Fallbacks for Missing Data**: Some queries include fallbacks if sheets are missing or empty
4. **Preserved Core Logic**: All business logic remains unchanged from the original queries

### Relationship Requirements (Unchanged):
- **Fact_Ticket_Summary[key] ↔ Fact_Status_Change[key]** (1:Many)
- **Fact_Ticket_Summary[CreatedDate] ↔ Dim_Date[Date]** (Many:1, Active)
- **Fact_Ticket_Summary[ResolvedDate] ↔ Dim_Date[Date]** (Many:1, Inactive)
- **Fact_Ticket_Summary[issue_type] ↔ Config_Issue_Type_Mapping[issue_type]** (Many:1)
- **Config_Issue_Type_Mapping[CapabilityKey] ↔ Dim_Capability[CapabilityKey]** (Many:1)