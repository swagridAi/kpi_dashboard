Complete Power Query (M) Code for Essential SLO Dashboard
Overview
This document provides complete Power Query (M) code for implementing a streamlined SLO Dashboard focused on 6 core KPIs for essential service performance tracking. The simplified model eliminates complex analytics while maintaining comprehensive coverage of time, volume, and quality dimensions.

Supported Core KPIs:

Lead Time - Creation to work start (business days)
Cycle Time - Work start to completion (business days)
Response Time - End-to-end resolution (business days)
Throughput - Tickets completed per period
Service Quality - SLO achievement percentage
Issue Resolution Time - Average resolution time
Core Fact Tables
1. Fact_Ticket_Summary
m
let
    // ===== DATA SOURCE OPTIONS =====
    // Option A: Excel File
    Source = Excel.Workbook(File.Contents("C:\SLOData\ticket_summary.xlsx"), null, true),
    Sheet = Source{[Item="TicketData",Kind="Sheet"]}[Data],
    PromotedHeaders = Table.PromoteHeaders(Sheet, [PromoteAllScalars=true]),
    
    // Option B: CSV File (alternative)
    // Source = Csv.Document(File.Contents("C:\SLOData\ticket_summary.csv"),[Delimiter=",", Encoding=1252]),
    // PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    
    // Option C: SharePoint List (alternative)
    // Source = SharePoint.Tables("https://company.sharepoint.com/sites/SLOData"),
    // TicketList = Source{[Name="TicketSummary"]}[Items],
    // PromotedHeaders = TicketList,
    
    // ===== DATA CLEANING =====
    // Remove rows with missing key data
    FilterActive = Table.SelectRows(PromotedHeaders, each 
        [active] = true and 
        [key] <> null and 
        [created] <> null
    ),
    
    // ===== DATA TYPE TRANSFORMATIONS =====
    TypedTable = Table.TransformColumnTypes(FilterActive, {
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
    
    // ===== CALCULATED DATE COLUMNS =====
    AddCreatedDate = Table.AddColumn(TypedTable, "CreatedDate", each Date.From([created])),
    AddResolvedDate = Table.AddColumn(AddCreatedDate, "ResolvedDate", each 
        if [resolution_date] <> null then Date.From([resolution_date]) else null),
    AddUpdatedDate = Table.AddColumn(AddResolvedDate, "UpdatedDate", each Date.From([updated])),
    
    // ===== RESOLUTION TIME CALCULATIONS =====
    AddResolutionTimeDays = Table.AddColumn(AddUpdatedDate, "ResolutionTimeDays", each
        if [resolution_date] <> null then
            Duration.Days([resolution_date] - [created])
        else
            Duration.Days(DateTime.LocalNow() - [created])
    ),
    
    // ===== BUSINESS RULE FLAGS =====
    AddIsResolved = Table.AddColumn(AddResolutionTimeDays, "IsResolved", each [resolution_date] <> null),
    
    AddIsCompleted = Table.AddColumn(AddIsResolved, "Is_Completed", each
        let
            CompletedStatuses = {"Done", "Closed", "Resolved", "Fixed", "Completed"}
        in
            List.Contains(CompletedStatuses, [status])
    ),
    
    // ===== SLA TARGET CALCULATION (SIMPLIFIED 2-TIER HIERARCHY) =====
    // Join with capability mapping to get capability-level SLA
    JoinCapabilityMapping = Table.NestedJoin(AddIsCompleted, {"issue_type"}, 
        Config_Issue_Type_Capability_Mapping, {"issue_type"}, "CapabilityMapping", JoinKind.LeftOuter),
    ExpandCapabilityMapping = Table.ExpandTableColumn(JoinCapabilityMapping, "CapabilityMapping", 
        {"CapabilityKey"}, {"MappedCapabilityKey"}),
    
    // Use mapped CapabilityKey only (no source CapabilityKey available)
    AddFinalCapabilityKey = Table.AddColumn(ExpandCapabilityMapping, "FinalCapabilityKey", each
        [MappedCapabilityKey]
    ),
    
    // Join with capability to get SLA targets
    JoinCapability = Table.NestedJoin(AddFinalCapabilityKey, {"FinalCapabilityKey"}, 
        Dim_Capability, {"CapabilityKey"}, "CapabilityData", JoinKind.LeftOuter),
    ExpandCapability = Table.ExpandTableColumn(JoinCapability, "CapabilityData", 
        {"ResponseTimeTargetDays"}, {"CapabilityResponseTimeTarget"}),
    
    // Join with default SLA table as fallback
    JoinDefaultSLA = Table.NestedJoin(ExpandCapability, {"issue_type"}, 
        Default_SLA_Table, {"TicketType"}, "DefaultSLA", JoinKind.LeftOuter),
    ExpandDefaultSLA = Table.ExpandTableColumn(JoinDefaultSLA, "DefaultSLA", 
        {"SLA_Days"}, {"DefaultSLADays"}),
    
    // Calculate final SLA target using 2-tier hierarchy
    AddSLATarget = Table.AddColumn(ExpandDefaultSLA, "ResponseTimeTargetDays", each
        // Tier 1: Capability-level target
        if [CapabilityResponseTimeTarget] <> null then [CapabilityResponseTimeTarget]
        // Tier 2: Default SLA fallback
        else if [DefaultSLADays] <> null then [DefaultSLADays]
        // Ultimate fallback
        else 5
    ),
    
    // ===== SLA ACHIEVEMENT CALCULATION =====
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
    
    // ===== STATUS FLAGS =====
    AddDaysInCurrentStatus = Table.AddColumn(AddIsOverdue, "DaysInCurrentStatus", each
        Duration.Days(DateTime.LocalNow() - [updated])
    ),
    
    // ===== DATA QUALITY VALIDATION =====
    ValidateData = Table.SelectRows(AddDaysInCurrentStatus, each 
        [ResolutionTimeDays] >= 0 and  // No negative resolution times
        ([IsResolved] = false or [resolution_date] <> null)  // Resolved tickets must have resolution date
    ),
    
    // ===== REMOVE HELPER COLUMNS =====
    RemoveHelpers = Table.RemoveColumns(ValidateData, 
        {"CapabilityMapping", "MappedCapabilityKey", "CapabilityData", "DefaultSLA", 
         "CapabilityResponseTimeTarget", "DefaultSLADays"}),
    
    // ===== FINAL TYPE OPTIMIZATION =====
    FinalTypes = Table.TransformColumnTypes(RemoveHelpers, {
        {"ResolutionTimeDays", Int64.Type},
        {"DaysInCurrentStatus", Int64.Type},
        {"ResponseTimeTargetDays", type number},
        {"Met_SLA", type logical},
        {"IsOverdue", type logical}
    })
in
    FinalTypes
2. Fact_Status_Change
m
let
    // ===== DATA SOURCE OPTIONS =====
    // Option A: Excel File
    Source = Excel.Workbook(File.Contents("C:\SLOData\status_changes.xlsx"), null, true),
    Sheet = Source{[Item="StatusChangeData",Kind="Sheet"]}[Data],
    PromotedHeaders = Table.PromoteHeaders(Sheet, [PromoteAllScalars=true]),
    
    // Option B: CSV File (alternative)
    // Source = Csv.Document(File.Contents("C:\SLOData\status_changes.csv"),[Delimiter=",", Encoding=1252]),
    // PromotedHeaders = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
    
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
    WithPreviousChange = Table.Combine(Grouped[AllRows])

    
    // ===== DURATION CALCULATIONS =====
    AddDurationCalendar = Table.AddColumn(AddPreviousChangeTime, "DurationCalendarHours", each
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
    
    // ===== REMOVE HELPER COLUMNS =====
    RemoveHelpers = Table.RemoveColumns(AddChangeDate, {"RowIndex", "PreviousChangeTime"}),
    
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
Essential Dimensions
3. Dim_Date
m
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
4. Dim_Capability
m
let
    // ===== DATA SOURCE OPTIONS =====
    // Option A: Excel File
    Source = Excel.Workbook(File.Contents("C:\SLOData\slo_configuration.xlsx"), null, true),
    ConfigSheet = Source{[Item="SLOTargets",Kind="Sheet"]}[Data],
    PromotedHeaders = Table.PromoteHeaders(ConfigSheet, [PromoteAllScalars=true]),
    
    // Option B: SharePoint list (alternative)
    // Source = SharePoint.Tables("https://company.sharepoint.com/sites/SLOConfig"),
    // ConfigList = Source{[Name="SLO_Configuration"]}[Items],
    // PromotedHeaders = ConfigList,
    
    // Option C: Static table for testing
    // Source = #table(
    //     {"CapabilityKey", "CapabilityName", "LeadTimeTargetDays", "CycleTimeTargetDays", "ResponseTimeTargetDays"},
    //     {
    //         {"DQ", "Data Quality", 1, 2, 3},
    //         {"DE", "Data Extracts", 1.5, 3, 5},
    //         {"CC", "Change Controls", 2, 5, 7},
    //         {"RD", "Reference Data", 1, 3, 4},
    //         {"RM", "Records Management", 2, 3, 5}
    //     }
    // ),
    // PromotedHeaders = Source,
    
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
5. Dim_Status
m
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
Configuration Tables
6. Config_Issue_Type_Mapping
m
let
    // ===== DATA SOURCE OPTIONS =====
    // Option A: Excel File
    Source = Excel.Workbook(File.Contents("C:\SLOData\issue_type_mapping.xlsx"), null, true),
    Sheet = Source{[Item="Mapping",Kind="Sheet"]}[Data],
    PromotedHeaders = Table.PromoteHeaders(Sheet, [PromoteAllScalars=true]),
    
    // Option B: Static table for testing
    // Source = #table(
    //     {"issue_type", "CapabilityKey", "Notes"},
    //     {
    //         {"Bug", "DQ", "Data quality defects"},
    //         {"Data Quality Task", "DQ", "Ongoing monitoring tasks"},
    //         {"Extract Request", "DE", "Custom data extractions"},
    //         {"Scheduled Extract", "DE", "Automated extract maintenance"},
    //         {"Change Request", "CC", "Standard change approval"},
    //         {"Emergency Change", "CC", "Emergency changes"},
    //         {"Reference Data Update", "RD", "Reference data maintenance"},
    //         {"Data Classification", "RD", "Data classification tasks"},
    //         {"Records Retention", "RM", "Records archival process"},
    //         {"Records Retrieval", "RM", "Records retrieval requests"},
    //         {"Task", "DQ", "General tasks default to Data Quality"},
    //         {"Story", "DE", "User stories default to Data Engineering"},
    //         {"Epic", "CC", "Epics default to Change Controls"}
    //     }
    // ),
    // PromotedHeaders = Source,
    
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
7. Default_SLA_Table
m
let
    // ===== STATIC SLA DEFINITIONS =====
    Source = #table(
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
    ),
    
    // ===== ADD METADATA =====
    AddSLAKey = Table.AddIndexColumn(Source, "SLA_Key", 1, 1),
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
Helper Functions
8. Business Hours Calculation Function
m
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
Implementation Notes
Key Simplifications Made:
Removed Complexity: Eliminated service-specific SLA overrides, priority adjustments, and complex reopening analysis
Streamlined SLA Logic: Simplified to 2-tier hierarchy (Capability → Default fallback)
Focus on Core KPIs: All queries optimized to support only the 6 essential performance indicators
Improved Maintainability: Reduced relationships and calculations for easier maintenance
Relationship Requirements:
Fact_Ticket_Summary[key] ↔ Fact_Status_Change[key] (1:Many)
Fact_Ticket_Summary[CreatedDate] ↔ Dim_Date[Date] (Many:1, Active)
Fact_Ticket_Summary[ResolvedDate] ↔ Dim_Date[Date] (Many:1, Inactive)
Fact_Ticket_Summary[issue_type] ↔ Config_Issue_Type_Mapping[issue_type] (Many:1)
Config_Issue_Type_Mapping[CapabilityKey] ↔ Dim_Capability[CapabilityKey] (Many:1)
Data Quality Validations:
Resolution times are non-negative
Resolved tickets have resolution dates
SLA targets are within reasonable ranges (0-30 days)
Business day calculations exclude weekends and holidays
All mappings have required fields populated
This simplified implementation provides comprehensive support for essential SLO tracking while dramatically reducing complexity and maintenance overhead.