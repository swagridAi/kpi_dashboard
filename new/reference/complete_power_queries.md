# Complete Power Query (M) Code for SLO Dashboard Tables

## Fact Tables

### 1. Fact_Ticket_Summary
```m
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
        [Active] = true and 
        [TicketKey] <> null and 
        [Created] <> null
    ),
    
    // ===== DATA TYPE TRANSFORMATIONS =====
    TypedTable = Table.TransformColumnTypes(FilterActive, {
        {"TicketKey", type text},
        {"IssueType", type text},
        {"Status", type text},
        {"Created", type datetime},
        {"Updated", type datetime},
        {"ResolutionDate", type datetime},
        {"Priority", type text},
        {"AssigneeDisplayName", type text},
        {"Summary", type text},
        {"Active", type logical},
        {"CapabilityKey", type text}
    }),
    
    // ===== CALCULATED DATE COLUMNS =====
    AddCreatedDate = Table.AddColumn(TypedTable, "CreatedDate", each Date.From([Created])),
    AddResolvedDate = Table.AddColumn(AddCreatedDate, "ResolvedDate", each 
        if [ResolutionDate] <> null then Date.From([ResolutionDate]) else null),
    AddUpdatedDate = Table.AddColumn(AddResolvedDate, "UpdatedDate", each Date.From([Updated])),
    
    // ===== RESOLUTION TIME CALCULATIONS =====
    AddResolutionTimeDays = Table.AddColumn(AddUpdatedDate, "ResolutionTimeDays", each
        if [ResolutionDate] <> null then
            Duration.Days([ResolutionDate] - [Created])
        else
            Duration.Days(DateTime.LocalNow() - [Created])
    ),
    
    // ===== BUSINESS RULE FLAGS =====
    AddIsResolved = Table.AddColumn(AddResolutionTimeDays, "IsResolved", each [ResolutionDate] <> null),
    
    AddIsCompleted = Table.AddColumn(AddIsResolved, "Is_Completed", each
        let
            CompletedStatuses = {"Done", "Closed", "Resolved", "Fixed", "Completed"}
        in
            List.Contains(CompletedStatuses, [Status])
    ),
    
    AddIsOverdue = Table.AddColumn(AddIsCompleted, "IsOverdue", each
        if [ResolutionDate] = null then
            [ResolutionTimeDays] > 5  // Default 5-day threshold
        else false
    ),
    
    // ===== STATUS FLAGS =====
    AddDaysInCurrentStatus = Table.AddColumn(AddIsOverdue, "DaysInCurrentStatus", each
        Duration.Days(DateTime.LocalNow() - [Updated])
    ),
    
    // ===== REOPENING DETECTION =====
    // This is a simplified version - full reopening logic would require status change history
    AddWasReopened = Table.AddColumn(AddDaysInCurrentStatus, "Was_Reopened", each
        // Simplified logic - can be enhanced with status change data
        if Text.Contains([Summary], "reopen", Comparer.OrdinalIgnoreCase) or 
           Text.Contains([Summary], "reopened", Comparer.OrdinalIgnoreCase) then true 
        else false
    ),
    
    AddReopenCount = Table.AddColumn(AddWasReopened, "Reopen_Count", each
        if [Was_Reopened] then 1 else 0
    ),
    
    // ===== DATA QUALITY VALIDATION =====
    ValidateData = Table.SelectRows(AddReopenCount, each 
        [ResolutionTimeDays] >= 0 and  // No negative resolution times
        ([IsResolved] = false or [ResolutionDate] <> null)  // Resolved tickets must have resolution date
    ),
    
    // ===== FINAL TYPE OPTIMIZATION =====
    FinalTypes = Table.TransformColumnTypes(ValidateData, {
        {"ResolutionTimeDays", Int64.Type},
        {"DaysInCurrentStatus", Int64.Type},
        {"Reopen_Count", Int64.Type}
    })
in
    FinalTypes
```

### 2. Fact_Status_Change
```m
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
        [Active] = true and 
        [TicketKey] <> null and 
        [ChangeCreated] <> null
    ),
    
    // ===== DATA TYPES =====
    TypedData = Table.TransformColumnTypes(FilterActive, {
        {"ChangeID", Int64.Type},
        {"TicketKey", type text},
        {"ChangeCreated", type datetime},
        {"FromStatus", type text},
        {"ToStatus", type text},
        {"AuthorDisplayName", type text},
        {"Active", type logical}
    }),
    
    // ===== SORT FOR DURATION CALCULATIONS =====
    SortedData = Table.Sort(TypedData, {{"TicketKey", Order.Ascending}, {"ChangeCreated", Order.Ascending}}),
    
    // ===== ADD INDEX FOR WINDOW FUNCTIONS =====
    AddIndex = Table.AddIndexColumn(SortedData, "RowIndex", 0),
    
    // ===== CALCULATE PREVIOUS CHANGE TIME =====
    AddPreviousChangeTime = Table.AddColumn(AddIndex, "PreviousChangeTime", (currentRow) =>
        let
            CurrentKey = currentRow[TicketKey],
            CurrentIndex = currentRow[RowIndex],
            PreviousRows = Table.SelectRows(SortedData, each 
                [TicketKey] = CurrentKey and [RowIndex] < CurrentIndex
            ),
            LastRow = if Table.RowCount(PreviousRows) > 0 then 
                Table.Last(PreviousRows)[ChangeCreated] 
            else 
                currentRow[ChangeCreated]  // First change for ticket
        in LastRow
    ),
    
    // ===== DURATION CALCULATIONS =====
    AddDurationCalendar = Table.AddColumn(AddPreviousChangeTime, "DurationCalendarHours", each
        Duration.TotalHours([ChangeCreated] - [PreviousChangeTime])
    ),
    
    // ===== BUSINESS HOURS CALCULATION =====
    AddDurationBusiness = Table.AddColumn(AddDurationCalendar, "DurationBusinessHours", each
        let
            StartTime = [PreviousChangeTime],
            EndTime = [ChangeCreated],
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
            FromBacklog = List.Contains(BacklogStatuses, [FromStatus]),
            ToActive = List.Contains(LeadTimeStatuses, [ToStatus])
        in FromBacklog and ToActive
    ),
    
    AddCycleTimeFlag = Table.AddColumn(AddLeadTimeFlag, "IsCycleTimeStart", each
        let
            CycleTimeStatuses = {"In Progress", "Development", "Testing", "Review"},
            ToActive = List.Contains(CycleTimeStatuses, [ToStatus]),
            FromInactive = not List.Contains(CycleTimeStatuses, [FromStatus])
        in ToActive and FromInactive
    ),
    
    AddResponseTimeFlag = Table.AddColumn(AddCycleTimeFlag, "IsResponseTimeEnd", each
        let
            DoneStatuses = {"Done", "Resolved", "Closed", "Completed"}
        in List.Contains(DoneStatuses, [ToStatus])
    ),
    
    // ===== REOPENING DETECTION =====
    AddReopenEvent = Table.AddColumn(AddResponseTimeFlag, "ReopenEvent", each
        let
            DoneStatuses = {"Done", "Resolved", "Closed", "Completed", "Fixed"},
            OpenStatuses = {"To Do", "Open", "In Progress", "Backlog", "New", "Reopened"},
            
            FromDone = List.Contains(DoneStatuses, [FromStatus]),
            ToOpen = List.Contains(OpenStatuses, [ToStatus]),
            
            // Exclude immediate corrections (< 30 minutes)
            NotImmediate = Duration.TotalMinutes([ChangeCreated] - [PreviousChangeTime]) > 30,
            
            // Was actually resolved for meaningful period (> 1 hour)
            WasResolved = Duration.TotalHours([ChangeCreated] - [PreviousChangeTime]) > 1
        in
            FromDone and ToOpen and NotImmediate and WasResolved
    ),
    
    // ===== ADD RELATIONSHIP KEYS =====
    AddChangeDate = Table.AddColumn(AddReopenEvent, "ChangeDate", each Date.From([ChangeCreated])),
    
    // ===== REMOVE HELPER COLUMNS =====
    RemoveHelpers = Table.RemoveColumns(AddChangeDate, {"RowIndex", "PreviousChangeTime"}),
    
    // ===== FINAL DATA TYPES =====
    FinalTypes = Table.TransformColumnTypes(RemoveHelpers, {
        {"DurationCalendarHours", type number},
        {"DurationBusinessHours", type number},
        {"IsLeadTimeStart", type logical},
        {"IsCycleTimeStart", type logical},
        {"IsResponseTimeEnd", type logical},
        {"ReopenEvent", type logical},
        {"ChangeDate", type date}
    })
in
    FinalTypes
```

## Dimension Tables

### 3. Dim_Date
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
    
    // ===== HOLIDAYS (Basic US Holidays) =====
    AddIsHoliday = Table.AddColumn(AddFiscalQuarter, "IsHoliday", each
        let
            Month = [Month],
            Day = Date.Day([Date]),
            Year = [Year]
        in
            // New Year's Day
            (Month = 1 and Day = 1) or
            // Independence Day
            (Month = 7 and Day = 4) or
            // Christmas Day
            (Month = 12 and Day = 25) or
            // Labor Day (First Monday in September)
            (Month = 9 and [DayOfWeek] = 1 and Day <= 7) or
            // Thanksgiving (4th Thursday in November)
            (Month = 11 and [DayOfWeek] = 4 and Day >= 22 and Day <= 28)
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

### 4. Dim_Capability
```m
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
        switch [CapabilityKey]
            case "DQ" then "Data Quality Team Lead"
            case "DE" then "Data Engineering Manager"
            case "CC" then "Change Control Board"
            case "RD" then "Data Architecture Team"
            case "RM" then "Information Governance"
            otherwise "TBD"
    ),
    
    AddBusinessDomain = Table.AddColumn(AddCapabilityOwner, "BusinessDomain", each
        switch [CapabilityKey]
            case "DQ" then "Data Management"
            case "DE" then "Data Engineering"
            case "CC" then "IT Operations"
            case "RD" then "Data Architecture"
            case "RM" then "Compliance"
            otherwise "General"
    ),
    
    AddMaturityLevel = Table.AddColumn(AddBusinessDomain, "MaturityLevel", each
        switch [CapabilityKey]
            case "DQ" then "Advanced"
            case "DE" then "Intermediate"
            case "CC" then "Advanced"
            case "RD" then "Intermediate"
            case "RM" then "Basic"
            otherwise "Basic"
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

### 5. Dim_Service (Optional - for service-level SLA overrides)
```m
let
    // ===== STATIC SERVICE DEFINITIONS =====
    Source = #table(
        {"ServiceKey", "ServiceName", "CapabilityKey", "AutomationLevel", "TypicalEffortHours", "ServiceResponseTimeTarget"},
        {
            {"DQ-VALIDATE", "Data Validation Rules", "DQ", "Semi-Automated", 4, 2},
            {"DQ-CLEANSE", "Data Cleansing", "DQ", "Manual", 8, 4},
            {"DQ-MONITOR", "Quality Monitoring", "DQ", "Automated", 2, 1},
            {"DE-CUSTOM", "Custom Data Extract", "DE", "Manual", 12, 7},
            {"DE-STANDARD", "Standard Extract", "DE", "Semi-Automated", 6, 3},
            {"DE-AUTOMATED", "Automated Extract", "DE", "Automated", 2, 1},
            {"CC-EMERGENCY", "Emergency Change", "CC", "Manual", 2, 0.5},
            {"CC-STANDARD", "Standard Change", "CC", "Semi-Automated", 8, 5},
            {"CC-NORMAL", "Normal Change", "CC", "Manual", 16, 10},
            {"RD-UPDATE", "Reference Data Update", "RD", "Manual", 4, 3},
            {"RD-CLASSIFY", "Data Classification", "RD", "Manual", 6, 5},
            {"RM-ARCHIVE", "Records Archival", "RM", "Semi-Automated", 3, 2},
            {"RM-RETRIEVE", "Records Retrieval", "RM", "Manual", 2, 1}
        }
    ),
    
    // ===== ADD METADATA =====
    AddIsActive = Table.AddColumn(Source, "IsActive", each true),
    AddDeliveryMethod = Table.AddColumn(AddIsActive, "DeliveryMethod", each
        switch [AutomationLevel]
            case "Automated" then "API"
            case "Semi-Automated" then "Portal"
            case "Manual" then "Email"
            otherwise "Portal"
    ),
    
    AddComplexityScore = Table.AddColumn(AddDeliveryMethod, "ComplexityScore", each
        switch true
            case [TypicalEffortHours] <= 2 then 1
            case [TypicalEffortHours] <= 6 then 2
            case [TypicalEffortHours] <= 12 then 3
            otherwise 4
    ),
    
    // ===== DATA TYPES =====
    TypedServices = Table.TransformColumnTypes(AddComplexityScore, {
        {"ServiceKey", type text},
        {"ServiceName", type text},
        {"CapabilityKey", type text},
        {"AutomationLevel", type text},
        {"TypicalEffortHours", type number},
        {"ServiceResponseTimeTarget", type number},
        {"IsActive", type logical},
        {"DeliveryMethod", type text},
        {"ComplexityScore", Int64.Type}
    })
in
    TypedServices
```

### 6. Dim_Status
```m
let
    // ===== STATIC STATUS DEFINITIONS =====
    Source = #table(
        {"Status", "StatusCategory", "StatusOrder", "IncludeInLeadTime", "IncludeInCycleTime", "IncludeInResponseTime"},
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
        {"Status", type text},
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

### 7. Dim_Priority
```m
let
    // ===== STATIC PRIORITY DEFINITIONS =====
    Source = #table(
        {"Priority", "PriorityLevel", "SLAMultiplier", "EscalationHours", "BusinessImpact"},
        {
            {"P1", 1, 0.5, 2, "Critical - System down or major functionality impaired"},
            {"Critical", 1, 0.5, 2, "Critical - System down or major functionality impaired"},
            {"High", 1, 0.5, 2, "Critical - System down or major functionality impaired"},
            {"P2", 2, 0.75, 8, "High - Significant impact on business operations"},
            {"Important", 2, 0.75, 8, "High - Significant impact on business operations"},
            {"P3", 3, 1.0, 24, "Medium - Moderate impact, workaround available"},
            {"Medium", 3, 1.0, 24, "Medium - Moderate impact, workaround available"},
            {"Normal", 3, 1.0, 24, "Medium - Moderate impact, workaround available"},
            {"P4", 4, 1.5, 72, "Low - Minor impact, can be addressed in next release"},
            {"Low", 4, 1.5, 72, "Low - Minor impact, can be addressed in next release"},
            {"Minor", 4, 1.5, 72, "Low - Minor impact, can be addressed in next release"}
        }
    ),
    
    // ===== ADD RESPONSE EXPECTATIONS =====
    AddResponseExpectation = Table.AddColumn(Source, "ResponseExpectation", each
        switch [PriorityLevel]
            case 1 then "Immediate response required, 24/7 support"
            case 2 then "Same business day response"
            case 3 then "Next business day response"
            case 4 then "Response within 3 business days"
            otherwise "Standard response time"
    ),
    
    AddNotificationLevel = Table.AddColumn(AddResponseExpectation, "NotificationLevel", each
        switch [PriorityLevel]
            case 1 then "Executive + Team + On-Call"
            case 2 then "Team Lead + Assignee"
            case 3 then "Assignee + Team"
            case 4 then "Assignee Only"
            otherwise "Standard"
    ),
    
    // ===== ADD METADATA =====
    AddPriorityKey = Table.AddIndexColumn(AddNotificationLevel, "PriorityKey", 1, 1),
    AddIsActive = Table.AddColumn(AddPriorityKey, "IsActive", each true),
    AddCreatedDate = Table.AddColumn(AddIsActive, "CreatedDate", each Date.From(DateTime.LocalNow())),
    
    // ===== DATA TYPES =====
    TypedPriorities = Table.TransformColumnTypes(AddCreatedDate, {
        {"PriorityKey", Int64.Type},
        {"Priority", type text},
        {"PriorityLevel", Int64.Type},
        {"SLAMultiplier", type number},
        {"EscalationHours", type number},
        {"BusinessImpact", type text},
        {"ResponseExpectation", type text},
        {"NotificationLevel", type text},
        {"IsActive", type logical},
        {"CreatedDate", type date}
    })
in
    TypedPriorities
```

### 8. Dim_Assignee
```m
let
    // ===== DATA SOURCE OPTIONS =====
    // Option A: Extract from ticket data
    TicketSource = Fact_Ticket_Summary,
    UniqueAssignees = Table.Distinct(TicketSource, {"AssigneeDisplayName"}),
    
    // Option B: Static table for testing
    // Source = #table(
    //     {"AssigneeDisplayName", "CapabilityKey", "TeamLead", "Department"},
    //     {
    //         {"John Doe", "DQ", false, "Data Quality"},
    //         {"Jane Smith", "DQ", true, "Data Quality"},
    //         {"Bob Johnson", "DE", false, "Data Engineering"},
    //         {"Alice Brown", "DE", true, "Data Engineering"},
    //         {"Charlie Wilson", "CC", false, "Change Management"},
    //         {"Diana Davis", "CC", true, "Change Management"}
    //     }
    // ),
    // UniqueAssignees = Source,
    
    // ===== ENRICH WITH METADATA =====
    AddAssigneeKey = Table.AddIndexColumn(UniqueAssignees, "AssigneeKey", 1, 1),
    
    AddEmailAddress = Table.AddColumn(AddAssigneeKey, "EmailAddress", each
        // Generate email from display name (for demo purposes)
        let
            CleanName = Text.Lower(Text.Replace([AssigneeDisplayName], " ", ".")),
            Email = CleanName & "@company.com"
        in Email
    ),
    
    // ===== ADD CAPABILITY MAPPING =====
    // Try to determine capability from ticket data
    AddCapabilityKey = Table.AddColumn(AddEmailAddress, "CapabilityKey", each
        let
            // Look up most common capability for this assignee
            AssigneeTickets = Table.SelectRows(TicketSource, each [AssigneeDisplayName] = [AssigneeDisplayName]),
            CapabilityCounts = Table.Group(AssigneeTickets, {"CapabilityKey"}, {{"Count", each Table.RowCount(_), Int64.Type}}),
            TopCapability = Table.Top(CapabilityCounts, 1)[CapabilityKey]{0}?
        in TopCapability ?? "Unknown"
    ),
    
    AddTeamLead = Table.AddColumn(AddCapabilityKey, "TeamLead", each
        // Simple rule: if name contains "Lead" or "Manager"
        Text.Contains([AssigneeDisplayName], "Lead") or 
        Text.Contains([AssigneeDisplayName], "Manager") or
        Text.Contains([AssigneeDisplayName], "Senior")
    ),
    
    AddDepartment = Table.AddColumn(AddTeamLead, "Department", each
        switch [CapabilityKey]
            case "DQ" then "Data Quality"
            case "DE" then "Data Engineering"
            case "CC" then "Change Management"
            case "RD" then "Data Architecture"
            case "RM" then "Information Governance"
            otherwise "General IT"
    ),
    
    AddManagerName = Table.AddColumn(AddDepartment, "ManagerName", each
        switch [CapabilityKey]
            case "DQ" then "Data Quality Manager"
            case "DE" then "Data Engineering Manager"
            case "CC" then "Change Control Board Chair"
            case "RD" then "Chief Data Architect"
            case "RM" then "Information Governance Manager"
            otherwise "IT Manager"
    ),
    
    // ===== ADD STATUS FLAGS =====
    AddIsActive = Table.AddColumn(AddManagerName, "IsActive", each true),
    AddCreatedDate = Table.AddColumn(AddIsActive, "CreatedDate", each Date.From(DateTime.LocalNow())),
    
    // ===== DATA TYPES =====
    TypedAssignees = Table.TransformColumnTypes(AddCreatedDate, {
        {"AssigneeKey", Int64.Type},
        {"AssigneeDisplayName", type text},
        {"EmailAddress", type text},
        {"CapabilityKey", type text},
        {"TeamLead", type logical},
        {"Department", type text},
        {"ManagerName", type text},
        {"IsActive", type logical},
        {"CreatedDate", type date}
    })
in
    TypedAssignees
```

## Configuration Tables

### 9. Config_Issue_Type_Mapping
```m
let
    // ===== DATA SOURCE OPTIONS =====
    // Option A: Excel File
    Source = Excel.Workbook(File.Contents("C:\SLOData\issue_type_mapping.xlsx"), null, true),
    Sheet = Source{[Item="Mapping",Kind="Sheet"]}[Data],
    PromotedHeaders = Table.PromoteHeaders(Sheet, [PromoteAllScalars=true]),
    
    // Option B: Static table for testing
    // Source = #table(
    //     {"IssueType", "CapabilityKey", "ServiceKey", "Notes"},
    //     {
    //         {"Bug", "DQ", "DQ-VALIDATE", "Data quality defects"},
    //         {"Data Quality Task", "DQ", "DQ-MONITOR", "Ongoing monitoring tasks"},
    //         {"Extract Request", "DE", "DE-CUSTOM", "Custom data extractions"},
    //         {"Scheduled Extract", "DE", "DE-AUTOMATED", "Automated extract maintenance"},
    //         {"Change Request", "CC", "CC-STANDARD", "Standard change approval"},
    //         {"Emergency Change", "CC", "CC-EMERGENCY", "Emergency changes"},
    //         {"Reference Data Update", "RD", "RD-UPDATE", "Reference data maintenance"},
    //         {"Data Classification", "RD", "RD-CLASSIFY", "Data classification tasks"},
    //         {"Records Retention", "RM", "RM-ARCHIVE", "Records archival process"},
    //         {"Records Retrieval", "RM", "RM-RETRIEVE", "Records retrieval requests"},
    //         {"Task", "DQ", null, "General tasks default to Data Quality"},
    //         {"Story", "DE", null, "User stories default to Data Engineering"},
    //         {"Epic", "CC", null, "Epics default to Change Controls"}
    //     }
    // ),
    // PromotedHeaders = Source,
    
    // ===== FILTER ACTIVE MAPPINGS =====
    FilterActive = Table.SelectRows(PromotedHeaders, each [IsActive] = true),
    
    // ===== DATA TYPES =====
    TypedMapping = Table.TransformColumnTypes(FilterActive, {
        {"IssueType", type text},
        {"CapabilityKey", type text},
        {"ServiceKey", type text},
        {"Notes", type text},
        {"IsActive", type logical}
    }),
    
    // ===== ADD VALIDATION =====
    AddMappingKey = Table.AddIndexColumn(TypedMapping, "MappingKey", 1, 1),
    AddIsDefault = Table.AddColumn(AddMappingKey, "IsDefault", each [ServiceKey] = null),
    AddEffectiveDate = Table.AddColumn(AddIsDefault, "EffectiveDate", each Date.From(DateTime.LocalNow())),
    AddCreatedBy = Table.AddColumn(AddEffectiveDate, "CreatedBy", each "System Administrator"),
    
    // ===== VALIDATION RULES =====
    ValidateMappings = Table.SelectRows(AddCreatedBy, each 
        [IssueType] <> null and 
        [CapabilityKey] <> null and
        Text.Length([IssueType]) > 0 and
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

### 10. Default_SLA_Table
```m
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
```

## Helper Functions (Optional)

### Business Hours Calculation Function
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
