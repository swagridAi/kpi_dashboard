Power BI Development Guide
Architecture Overview
The simplified SLO Dashboard uses a star schema dimensional model optimized for Power BI's VertiPaq engine. The architecture focuses on essential service delivery metrics through 6 core KPIs, balancing performance with maintainability.

Dimensional Model Design
Core Design Principles:

Star schema with centralized fact tables and essential dimensions
Pre-calculated business logic in Power Query for optimal performance
Active/inactive relationships enabling basic time intelligence
Memory-optimized data types and compression strategies
Key Components:

2 Fact Tables: Fact_Ticket_Summary (tickets), Fact_Ticket_Status_Change (status transitions)
3 Core Dimensions: Dim_Date, Dim_Status, Dim_Capability
2 Configuration Tables: Config_Issue_Type_Capability_Mapping, Default_SLA_Table
Simplified Star Schema Implementation
mermaid
erDiagram
    FACT_TICKET_STATUS_CHANGE ||--o{ DIM_DATE : "ChangeDate (Active)"
    FACT_TICKET_STATUS_CHANGE ||--o{ DIM_STATUS : "from_string (Active)"
    FACT_TICKET_STATUS_CHANGE ||--o{ DIM_STATUS : "to_string (Inactive)"
    FACT_TICKET_STATUS_CHANGE ||--o{ FACT_TICKET_SUMMARY : "TicketKey → key"
    
    FACT_TICKET_SUMMARY ||--o{ DIM_DATE : "CreatedDate (Active)"
    FACT_TICKET_SUMMARY ||--o{ DIM_DATE : "ResolvedDate (Inactive)"
    FACT_TICKET_SUMMARY ||--o{ CONFIG_ISSUE_TYPE_MAPPING : "issue_type"
    FACT_TICKET_SUMMARY ||--o{ DIM_STATUS : "status"
    
    CONFIG_ISSUE_TYPE_MAPPING ||--o{ DIM_CAPABILITY : "CapabilityKey"
    FACT_TICKET_SUMMARY ||--o{ DEFAULT_SLA_TABLE : "issue_type → TicketType (Inactive)"
Relationship Cardinalities:

All fact-to-dimension relationships: Many-to-One
Configuration mappings: Many-to-One
Time intelligence: Many-to-One with active/inactive relationships
Data Model Implementation
Fact Tables
Fact_Ticket_Summary
Purpose: Core ticket tracking with simplified SLO calculations

Complete Power Query Implementation:

m
// Simplified Fact_Ticket_Summary focused on 6 core KPIs
let
    // Source data with initial filtering
    Source = Table.SelectRows(Source_jira_snapshot, each [active] = 1),
    
    // Essential data types
    TypedTable = Table.TransformColumnTypes(Source, {
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
    
    // Date columns for relationships
    AddCreatedDate = Table.AddColumn(TypedTable, "CreatedDate", each Date.From([created])),
    AddResolvedDate = Table.AddColumn(AddCreatedDate, "ResolvedDate", each 
        if [resolution_date] <> null then Date.From([resolution_date]) else null),
    
    // Core business calculations
    AddResolutionTimeDays = Table.AddColumn(AddResolvedDate, "ResolutionTimeDays", each
        if [resolution_date] <> null then
            Duration.Days([resolution_date] - [created])
        else
            Duration.Days(DateTime.LocalNow() - [created])
    ),
    
    // Essential business flags
    AddIsResolved = Table.AddColumn(AddResolutionTimeDays, "IsResolved", each [resolution_date] <> null),
    
    AddIsCompleted = Table.AddColumn(AddIsResolved, "Is_Completed", each
        let
            CompletedStatuses = {"Done", "Closed", "Resolved", "Fixed", "Completed"}
        in
            List.Contains(CompletedStatuses, [status])
    ),
    
    // Get SLO targets through simplified 2-tier hierarchy
    JoinCapability = Table.NestedJoin(AddIsCompleted, {"issue_type"}, 
        Config_Issue_Type_Capability_Mapping, {"IssueType"}, "CapabilityMapping", JoinKind.LeftOuter),
    ExpandCapability = Table.ExpandTableColumn(JoinCapability, "CapabilityMapping", 
        {"CapabilityKey"}, {"CapabilityKey"}),
    
    JoinSLOTargets = Table.NestedJoin(ExpandCapability, {"CapabilityKey"}, 
        Dim_Capability, {"CapabilityKey"}, "SLOTargets", JoinKind.LeftOuter),
    ExpandSLOTargets = Table.ExpandTableColumn(JoinSLOTargets, "SLOTargets", 
        {"ResponseTimeTargetDays"}, {"ResponseTimeTargetDays"}),
    
    // Simplified SLA achievement calculation
    AddMetSLA = Table.AddColumn(ExpandSLOTargets, "Met_SLA", each
        if [ResolutionTimeDays] <> null and [ResponseTimeTargetDays] <> null then
            [ResolutionTimeDays] <= [ResponseTimeTargetDays]
        else null
    ),
    
    // Final data type optimization
    TypedFinal = Table.TransformColumnTypes(AddMetSLA, {
        {"ResolutionTimeDays", Int64.Type},
        {"ResponseTimeTargetDays", type number},
        {"Met_SLA", type logical},
        {"IsResolved", type logical},
        {"Is_Completed", type logical},
        {"CreatedDate", type date},
        {"ResolvedDate", type date}
    }),
    
    // Remove helper columns
    FinalTable = Table.RemoveColumns(TypedFinal, {"CapabilityMapping", "SLOTargets"})
in
    FinalTable
Fact_Ticket_Status_Change
Purpose: Essential status transitions for lead/cycle/response time calculations

Complete Power Query Implementation:

m
// Simplified Fact_Ticket_Status_Change for basic timing
let
    // Source data filtering
    Source = Table.SelectRows(Source_jira_changelog, each [field] = "status" and [active] = 1),
    
    // Basic data types
    TypedData = Table.TransformColumnTypes(Source, {
        {"id", Int64.Type},
        {"key", type text},
        {"change_created", type datetime},
        {"from_string", type text},
        {"to_string", type text},
        {"active", type logical}
    }),
    
    // Sort for duration calculations
    SortedData = Table.Sort(TypedData, {{"key", Order.Ascending}, {"change_created", Order.Ascending}}),
    AddIndex = Table.AddIndexColumn(SortedData, "RowIndex", 0),
    
    // Calculate previous change time
    AddPreviousChangeTime = Table.AddColumn(AddIndex, "PreviousChangeTime", (currentRow) =>
        let
            CurrentKey = currentRow[key],
            CurrentIndex = currentRow[RowIndex],
            PreviousRows = Table.SelectRows(SortedData, each 
                [key] = CurrentKey and [RowIndex] < CurrentIndex
            ),
            LastRow = if Table.RowCount(PreviousRows) > 0 then 
                Table.Last(PreviousRows)[change_created] 
            else 
                currentRow[change_created]
        in LastRow
    ),
    
    // Basic duration calculations
    AddDurationCalendar = Table.AddColumn(AddPreviousChangeTime, "DurationCalendarHours", each
        Duration.TotalHours([change_created] - [PreviousChangeTime])
    ),
    
    // Business hours calculation (9 AM - 5 PM, weekdays only)
    AddDurationBusiness = Table.AddColumn(AddDurationCalendar, "DurationBusinessHours", each
        let
            StartTime = [PreviousChangeTime],
            EndTime = [change_created],
            StartDate = Date.From(StartTime),
            EndDate = Date.From(EndTime),
            
            // Generate list of business days
            DateList = List.Dates(StartDate, Duration.Days(EndDate - StartDate) + 1, #duration(1,0,0,0)),
            BusinessDays = List.Select(DateList, each Date.DayOfWeek(_, Day.Monday) < 5),
            
            // Calculate business hours for each day
            BusinessHours = List.Accumulate(BusinessDays, 0, (total, current) =>
                if current = StartDate and current = EndDate then
                    // Same day calculation
                    let
                        StartHour = Number.Max(Time.Hour(DateTime.Time(StartTime)), 9),
                        EndHour = Number.Min(Time.Hour(DateTime.Time(EndTime)), 17),
                        Hours = Number.Max(EndHour - StartHour, 0)
                    in total + Hours
                else if current = StartDate then
                    // First day
                    let
                        StartHour = Number.Max(Time.Hour(DateTime.Time(StartTime)), 9),
                        Hours = Number.Max(17 - StartHour, 0)
                    in total + Hours
                else if current = EndDate then
                    // Last day
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
    
    // Essential SLO timing flags only
    AddLeadTimeFlag = Table.AddColumn(AddDurationBusiness, "IsLeadTimeStart", each
        let
            BacklogStatuses = {"Backlog", "New", "Open", "To Do"},
            ActiveStatuses = {"In Progress", "Development", "Analysis", "Working"},
            FromBacklog = List.Contains(BacklogStatuses, [from_string]),
            ToActive = List.Contains(ActiveStatuses, [to_string])
        in FromBacklog and ToActive
    ),
    
    AddCycleTimeFlag = Table.AddColumn(AddLeadTimeFlag, "IsCycleTimeStart", each
        let
            ActiveStatuses = {"In Progress", "Development", "Testing", "Review"},
            ToActive = List.Contains(ActiveStatuses, [to_string]),
            FromInactive = not List.Contains(ActiveStatuses, [from_string])
        in ToActive and FromInactive
    ),
    
    AddResponseTimeFlag = Table.AddColumn(AddCycleTimeFlag, "IsResponseTimeEnd", each
        let
            DoneStatuses = {"Done", "Resolved", "Closed", "Completed"}
        in List.Contains(DoneStatuses, [to_string])
    ),
    
    // Add relationship keys
    AddTicketKey = Table.AddColumn(AddResponseTimeFlag, "TicketKey", each [key]),
    AddChangeDate = Table.AddColumn(AddTicketKey, "ChangeDate", each Date.From([change_created])),
    
    // Remove helper columns
    RemoveHelpers = Table.RemoveColumns(AddChangeDate, {"RowIndex", "PreviousChangeTime"}),
    
    // Final data types
    TypedFinal = Table.TransformColumnTypes(RemoveHelpers, {
        {"DurationCalendarHours", type number},
        {"DurationBusinessHours", type number},
        {"IsLeadTimeStart", type logical},
        {"IsCycleTimeStart", type logical},
        {"IsResponseTimeEnd", type logical},
        {"TicketKey", type text},
        {"ChangeDate", type date}
    })
in
    TypedFinal
Core Dimension Tables
Dim_Date
Purpose: Business calendar supporting time intelligence

Power Query Implementation:

m
// Essential date dimension for 6 core KPIs
let
    // Generate date range
    StartDate = #date(2023, 1, 1),
    EndDate = #date(2027, 12, 31),
    DateList = List.Dates(StartDate, Duration.Days(EndDate - StartDate) + 1, #duration(1,0,0,0)),
    DateTable = Table.FromList(DateList, Splitter.SplitByNothing(), {"Date"}),
    
    // Calendar hierarchy
    AddYear = Table.AddColumn(DateTable, "Year", each Date.Year([Date])),
    AddMonth = Table.AddColumn(AddYear, "Month", each Date.Month([Date])),
    AddQuarter = Table.AddColumn(AddMonth, "Quarter", each Date.QuarterOfYear([Date])),
    AddMonthName = Table.AddColumn(AddQuarter, "MonthName", each Date.MonthName([Date])),
    
    // Business day calculations
    AddDayOfWeek = Table.AddColumn(AddMonthName, "DayOfWeek", each Date.DayOfWeek([Date], Day.Monday) + 1),
    AddIsBusinessDay = Table.AddColumn(AddDayOfWeek, "IsBusinessDay", each [DayOfWeek] <= 5),
    
    // Period boundaries for aggregations
    AddMonthStart = Table.AddColumn(AddIsBusinessDay, "MonthStart", each Date.StartOfMonth([Date])),
    AddMonthEnd = Table.AddColumn(AddMonthStart, "MonthEnd", each Date.EndOfMonth([Date])),
    
    // Current period flags
    AddIsToday = Table.AddColumn(AddMonthEnd, "IsToday", each [Date] = Date.From(DateTime.LocalNow())),
    AddIsYTD = Table.AddColumn(AddIsToday, "IsYTD", each 
        [Year] = Date.Year(DateTime.LocalNow()) and [Date] <= Date.From(DateTime.LocalNow())),
    
    // Data type optimization
    TypedTable = Table.TransformColumnTypes(AddIsYTD, {
        {"Date", type date},
        {"Year", Int64.Type},
        {"Month", Int64.Type},
        {"Quarter", Int64.Type},
        {"MonthName", type text},
        {"DayOfWeek", Int64.Type},
        {"IsBusinessDay", type logical},
        {"MonthStart", type date},
        {"MonthEnd", type date},
        {"IsToday", type logical},
        {"IsYTD", type logical}
    })
in
    TypedTable
Dim_Capability
Purpose: Capability definitions with simplified SLO targets

Power Query Implementation:

m
// Simplified capability dimension
let
    // Source can be Excel, SharePoint, or static table
    Source = #table(
        {"CapabilityKey", "CapabilityName", "ResponseTimeTargetDays"},
        {
            {"DQ", "Data Quality", 3},
            {"DE", "Data Extracts", 5},
            {"CC", "Change Controls", 7},
            {"RD", "Reference Data", 4},
            {"RM", "Records Management", 5}
        }
    ),
    
    // Add descriptive information
    AddCapabilityOwner = Table.AddColumn(Source, "CapabilityOwner", each
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
    
    // Add metadata
    AddIsActive = Table.AddColumn(AddBusinessDomain, "IsActive", each true),
    AddCreatedDate = Table.AddColumn(AddIsActive, "CreatedDate", each Date.From(DateTime.LocalNow())),
    
    // Data type optimization
    TypedTable = Table.TransformColumnTypes(AddCreatedDate, {
        {"CapabilityKey", type text},
        {"CapabilityName", type text},
        {"ResponseTimeTargetDays", type number},
        {"CapabilityOwner", type text},
        {"BusinessDomain", type text},
        {"IsActive", type logical},
        {"CreatedDate", type date}
    })
in
    TypedTable
Dim_Status
Purpose: Status definitions for time calculations

Power Query Implementation:

m
// Essential status dimension
let
    Source = #table(
        {"Status", "StatusCategory", "IncludeInLeadTime", "IncludeInCycleTime", "IncludeInResponseTime"},
        {
            {"Backlog", "To Do", true, false, false},
            {"New", "To Do", true, false, false},
            {"Open", "To Do", true, false, false},
            {"To Do", "To Do", true, false, false},
            {"In Progress", "In Progress", false, true, false},
            {"Development", "In Progress", false, true, false},
            {"Analysis", "In Progress", false, true, false},
            {"Testing", "In Progress", false, true, false},
            {"Review", "In Progress", false, true, false},
            {"Waiting", "Waiting", false, false, false},
            {"Blocked", "Waiting", false, false, false},
            {"Done", "Done", false, false, true},
            {"Resolved", "Done", false, false, true},
            {"Closed", "Done", false, false, true},
            {"Completed", "Done", false, false, true}
        }
    ),
    
    // Add business logic
    AddTimeType = Table.AddColumn(Source, "TimeType", each
        if [IncludeInLeadTime] then "Lead"
        else if [IncludeInCycleTime] then "Cycle"
        else if [IncludeInResponseTime] then "Response"
        else "Other"
    ),
    
    AddIsActive = Table.AddColumn(AddTimeType, "IsActive", each true),
    
    // Data type optimization
    TypedTable = Table.TransformColumnTypes(AddIsActive, {
        {"Status", type text},
        {"StatusCategory", type text},
        {"IncludeInLeadTime", type logical},
        {"IncludeInCycleTime", type logical},
        {"IncludeInResponseTime", type logical},
        {"TimeType", type text},
        {"IsActive", type logical}
    })
in
    TypedTable
Configuration Tables
Config_Issue_Type_Capability_Mapping
Purpose: Maps Jira issue types to business capabilities

Power Query Implementation:

m
// Simplified issue type mapping
let
    Source = #table(
        {"IssueType", "CapabilityKey", "Notes"},
        {
            {"Bug", "DQ", "Data quality defects"},
            {"Data Quality Task", "DQ", "Quality monitoring tasks"},
            {"Extract Request", "DE", "Data extraction requests"},
            {"Scheduled Extract", "DE", "Automated extract maintenance"},
            {"Change Request", "CC", "Standard change approval"},
            {"Emergency Change", "CC", "Urgent changes"},
            {"Reference Data Update", "RD", "Reference data maintenance"},
            {"Data Classification", "RD", "Data classification tasks"},
            {"Records Retention", "RM", "Records archival process"},
            {"Records Retrieval", "RM", "Records retrieval requests"},
            {"Task", "DQ", "General tasks default to Data Quality"},
            {"Story", "DE", "User stories default to Data Engineering"},
            {"Epic", "CC", "Epics default to Change Controls"}
        }
    ),
    
    // Add metadata
    AddMappingKey = Table.AddIndexColumn(Source, "MappingKey", 1, 1),
    AddIsActive = Table.AddColumn(AddMappingKey, "IsActive", each true),
    AddCreatedDate = Table.AddColumn(AddIsActive, "CreatedDate", each Date.From(DateTime.LocalNow())),
    
    // Data type optimization
    TypedTable = Table.TransformColumnTypes(AddCreatedDate, {
        {"MappingKey", Int64.Type},
        {"IssueType", type text},
        {"CapabilityKey", type text},
        {"Notes", type text},
        {"IsActive", type logical},
        {"CreatedDate", type date}
    })
in
    TypedTable
Default_SLA_Table
Purpose: Fallback SLA definitions

Power Query Implementation:

m
// Simplified default SLA table
let
    Source = #table(
        {"TicketType", "SLA_Days", "ExcludeWeekends", "Notes"},
        {
            {"Bug", 3, true, "Critical defects require faster response"},
            {"Task", 5, true, "Standard business day response target"},
            {"Epic", 10, true, "Large initiatives allow longer response time"},
            {"Story", 8, true, "User stories standard processing"},
            {"Sub-task", 2, true, "Sub-tasks require quick turnaround"},
            {"Improvement", 7, true, "Enhancement requests standard timeline"},
            {"New Feature", 15, true, "New features require extended analysis"},
            {"Change Request", 10, true, "Change management standard process"},
            {"Incident", 1, false, "Production incidents have highest priority"},
            {"Service Request", 5, true, "Standard service delivery"}
        }
    ),
    
    // Add metadata
    AddSLAKey = Table.AddIndexColumn(Source, "SLA_Key", 1, 1),
    AddIsActive = Table.AddColumn(AddSLAKey, "IsActive", each true),
    AddCreatedDate = Table.AddColumn(AddIsActive, "CreatedDate", each Date.From(DateTime.LocalNow())),
    
    // Data type optimization
    TypedTable = Table.TransformColumnTypes(AddCreatedDate, {
        {"SLA_Key", Int64.Type},
        {"TicketType", type text},
        {"SLA_Days", type number},
        {"ExcludeWeekends", type logical},
        {"Notes", type text},
        {"IsActive", type logical},
        {"CreatedDate", type date}
    })
in
    TypedTable
Relationship Configuration
Complete Relationship Setup:

javascript
// Active Relationships (Primary navigation paths)
Model.Relationships.add({
    name: "TicketSummary_CreatedDate_Date",
    from: "Fact_Ticket_Summary[CreatedDate]",
    to: "Dim_Date[Date]",
    cardinality: "ManyToOne",
    crossFilterDirection: "Both",
    isActive: true
});

Model.Relationships.add({
    name: "StatusChange_ChangeDate_Date", 
    from: "Fact_Ticket_Status_Change[ChangeDate]",
    to: "Dim_Date[Date]",
    cardinality: "ManyToOne",
    crossFilterDirection: "Both",
    isActive: true
});

Model.Relationships.add({
    name: "TicketSummary_IssueType_Mapping",
    from: "Fact_Ticket_Summary[issue_type]",
    to: "Config_Issue_Type_Capability_Mapping[IssueType]",
    cardinality: "ManyToOne",
    crossFilterDirection: "Both",
    isActive: true
});

Model.Relationships.add({
    name: "Mapping_Capability",
    from: "Config_Issue_Type_Capability_Mapping[CapabilityKey]",
    to: "Dim_Capability[CapabilityKey]",
    cardinality: "ManyToOne",
    crossFilterDirection: "Both", 
    isActive: true
});

Model.Relationships.add({
    name: "StatusChange_TicketKey_Summary",
    from: "Fact_Ticket_Status_Change[TicketKey]",
    to: "Fact_Ticket_Summary[key]",
    cardinality: "ManyToOne",
    crossFilterDirection: "Both",
    isActive: true
});

// Inactive Relationships (For role-playing dimensions)
Model.Relationships.add({
    name: "TicketSummary_ResolvedDate_Date",
    from: "Fact_Ticket_Summary[ResolvedDate]",
    to: "Dim_Date[Date]",
    cardinality: "ManyToOne",
    crossFilterDirection: "Both",
    isActive: false
});

Model.Relationships.add({
    name: "StatusChange_FromStatus_Status",
    from: "Fact_Ticket_Status_Change[from_string]",
    to: "Dim_Status[status]",
    cardinality: "ManyToOne",
    crossFilterDirection: "Both",
    isActive: true
});

Model.Relationships.add({
    name: "StatusChange_ToStatus_Status",
    from: "Fact_Ticket_Status_Change[to_string]",
    to: "Dim_Status[status]",
    cardinality: "ManyToOne", 
    crossFilterDirection: "Both",
    isActive: false
});

Model.Relationships.add({
    name: "TicketSummary_DefaultSLA",
    from: "Fact_Ticket_Summary[issue_type]",
    to: "Default_SLA_Table[TicketType]",
    cardinality: "ManyToOne",
    crossFilterDirection: "Both",
    isActive: false
});
Core Business Logic
Simplified SLA Calculation
Two-Tier Hierarchy:

dax
// Simplified SLA target resolution
SLA_Target_Days = 
COALESCE(
    // Priority 1: Capability-level target
    RELATED(Dim_Capability[ResponseTimeTargetDays]),
    
    // Priority 2: Default SLA fallback
    CALCULATE(
        MAX(Default_SLA_Table[SLA_Days]),
        USERELATIONSHIP(Fact_Ticket_Summary[issue_type], Default_SLA_Table[TicketType]),
        Default_SLA_Table[IsActive] = TRUE
    ),
    
    // Priority 3: Ultimate fallback
    5
)
Business Day Calculations
Standard Business Rules:

Business Hours: 9:00 AM to 5:00 PM (local time)
Weekends: Excluded by default
Holidays: Corporate holiday calendar integration (optional)
Time Zone: UTC standardization with local display
Quality Metrics
Service Quality Definition:

Percentage of resolved tickets meeting SLA targets
Calculated as: (Tickets meeting SLA ÷ Total resolved tickets) × 100
No dependency on reopening logic (simplified approach)
Core DAX Calculations (6 KPIs)
1. Lead Time Average
dax
Lead_Time_Days = 
CALCULATE(
    AVERAGE(Fact_Ticket_Status_Change[DurationBusinessHours]),
    Fact_Ticket_Status_Change[IsLeadTimeStart] = TRUE
) / 24
2. Cycle Time Average
dax
Cycle_Time_Days = 
VAR CycleTimeTickets = 
    SUMMARIZE(
        FILTER(Fact_Ticket_Status_Change, [IsCycleTimeStart] = TRUE),
        [TicketKey],
        "CycleTime", SUM(Fact_Ticket_Status_Change[DurationBusinessHours])
    )
RETURN AVERAGEX(CycleTimeTickets, [CycleTime]) / 24
3. Response Time Average
dax
Response_Time_Days = 
CALCULATE(
    AVERAGE(Fact_Ticket_Summary[ResolutionTimeDays]),
    Fact_Ticket_Summary[IsResolved] = TRUE
)
4. Throughput
dax
Throughput = 
CALCULATE(
    COUNTROWS(Fact_Ticket_Summary),
    Fact_Ticket_Summary[Is_Completed] = TRUE,
    USERELATIONSHIP(Fact_Ticket_Summary[ResolvedDate], Dim_Date[Date])
)
5. Service Quality
dax
Service_Quality_Percentage = 
VAR ResolvedTickets = 
    FILTER(
        Fact_Ticket_Summary,
        [IsResolved] = TRUE && NOT ISBLANK([Met_SLA])
    )
VAR TicketsMetSLA = 
    FILTER(ResolvedTickets, [Met_SLA] = TRUE)
RETURN
    DIVIDE(COUNTROWS(TicketsMetSLA), COUNTROWS(ResolvedTickets), 0) * 100
6. Issue Resolution Time
dax
Issue_Resolution_Time_Days = 
CALCULATE(
    AVERAGE(Fact_Ticket_Summary[ResolutionTimeDays]),
    Fact_Ticket_Summary[IsResolved] = TRUE
)
Basic Time Intelligence
Month-over-Month Change:

dax
MoM_Change = 
VAR CurrentPeriod = [Service_Quality_Percentage]
VAR PreviousPeriod = 
    CALCULATE(
        [Service_Quality_Percentage],
        DATEADD(Dim_Date[Date], -1, MONTH)
    )
RETURN 
    IF(
        NOT ISBLANK(PreviousPeriod) && PreviousPeriod <> 0,
        CurrentPeriod - PreviousPeriod,
        BLANK()
    )
Rolling Three-Month Average:

dax
Three_Month_Average = 
CALCULATE(
    [Service_Quality_Percentage],
    DATESINPERIOD(Dim_Date[Date], MAX(Dim_Date[Date]), -3, MONTH)
)
Tickets at Risk:

dax
Tickets_At_Risk = 
VAR RiskThreshold = 0.8  -- 80% of SLA target
RETURN
COUNTROWS(
    FILTER(
        Fact_Ticket_Summary,
        [IsResolved] = FALSE &&
        [ResolutionTimeDays] >= [SLA_Target_Days] * RiskThreshold
    )
)
Dashboard Development
Essential Dashboard Design
Core Visualizations:

KPI Cards: 6 core metrics with trend indicators
Time Series Charts: Monthly trends for each KPI
Capability Comparison: Side-by-side performance
Ticket Volume: Throughput over time
At-Risk Tickets: List with countdown to SLA breach
Design Principles:

Mobile-first responsive design
Clear hierarchy with executive summary at top
Drill-down capability from summary to detail
Consistent color coding across all visuals
Visual Specifications
Color Standards:

dax
// Color definitions for consistency
Performance_Color = 
SWITCH(
    TRUE(),
    [Service_Quality_Percentage] >= 95, "#2E7D32",  // Green
    [Service_Quality_Percentage] >= 85, "#F57C00",  // Yellow
    [Service_Quality_Percentage] > 0, "#C62828",    // Red
    "#757575"                                       // Gray
)
Conditional Formatting:

Green: 95%+ SLA achievement
Yellow: 85-94% SLA achievement
Red: <85% SLA achievement
Gray: No data/insufficient data
Performance Optimization
Memory Management
Data Model Optimization:

Remove unused columns in Power Query
Use appropriate data types (integers vs. decimals)
Implement star schema strictly (no many-to-many)
Compress text columns with limited cardinality
Query Optimization:

dax
// Use variables to avoid recalculation
Optimized_KPI_Calculation = 
VAR FilteredTickets = 
    FILTER(
        Fact_Ticket_Summary,
        [IsResolved] = TRUE &&
        [ResolutionTimeDays] >= 0
    )
VAR TotalTickets = COUNTROWS(FilteredTickets)
VAR MetSLATickets = 
    COUNTROWS(FILTER(FilteredTickets, [Met_SLA] = TRUE))
RETURN DIVIDE(MetSLATickets, TotalTickets, 0) * 100
Incremental Refresh
Configuration for Fact Tables:

json
{
  "incrementalRefresh": {
    "pollingExpression": "DateTime.LocalNow()",
    "refreshPeriods": {
      "refreshPeriod": "1 month",
      "archivePeriod": "6 months"
    },
    "detectDataChanges": true
  }
}
Testing and Validation
Essential Test Cases
Data Accuracy Tests:

dax
// Validate SLO calculations
SLO_Calculation_Test = 
VAR TestTickets = 
    FILTER(
        Fact_Ticket_Summary,
        [key] IN {"TEST-001", "TEST-002", "TEST-003"}
    )
VAR ValidationResults = 
    ADDCOLUMNS(
        TestTickets,
        "Expected_Met_SLA", 
            SWITCH([key], 
                "TEST-001", TRUE,   // 2 days, target 3 days
                "TEST-002", FALSE,  // 6 days, target 5 days
                "TEST-003", TRUE,   // 1 day, target 3 days
                BLANK()
            ),
        "Actual_Met_SLA", [Met_SLA],
        "Test_Passed", [Expected_Met_SLA] = [Met_SLA]
    )
RETURN ValidationResults
Performance Validation:

dax
// Monitor calculation performance
Performance_Check = 
VAR StartTime = NOW()
VAR TestResult = [Service_Quality_Percentage]
VAR EndTime = NOW()
VAR ExecutionSeconds = DATEDIFF(StartTime, EndTime, SECOND)
RETURN 
    "Result: " & FORMAT(TestResult, "0.0%") & 
    " | Execution: " & ExecutionSeconds & "s"
Business Validation
Monthly Review Checklist:

 All 6 KPIs calculate correctly
 Business day logic excludes weekends appropriately
 SLA targets match capability configurations
 Ticket volumes align with business expectations
 Performance meets <3 second response time target
Deployment
Environment Strategy
Three-Tier Deployment:

Development: Feature development and initial testing
Staging: User acceptance testing and validation
Production: Live system with full access controls
Deployment Process:

Export development .pbix file
Deploy to staging workspace for validation
Update data source connections for each environment
Configure incremental refresh in production
Apply row-level security and permissions
Validate all calculations post-deployment
Version Control
Power BI Git Integration:

Track model changes with deployment pipelines
Document all major changes in deployment notes
Maintain rollback capability to previous versions
Automate deployment through Azure DevOps integration
This simplified implementation focuses exclusively on the 6 core KPIs while maintaining the technical rigor needed for enterprise deployment. The reduced complexity improves maintainability and performance while delivering essential service delivery insights.

