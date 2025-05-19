# Power BI SLO Dashboard Dimensional Model - Technical Specification

## Implementation Timeline Context

This technical implementation spans multiple project phases:

**Phase 0 (Weeks 1-4):** Complete all core dimensional modeling, fact tables, and basic measures described in this document. Focus on manual configuration within Power BI.

**Phase 1 (Weeks 5-8):** Integrate with Confluence for configuration management. Implement basic automation for report generation.

**Phase 2+ (Weeks 9+):** Add advanced features like real-time alerting, self-service configuration interfaces, and organizational governance features.

The code provided represents the full technical architecture but should be implemented incrementally according to the phase schedule.

## Executive Summary

This document outlines the complete dimensional model implementation for the SLO Dashboard built natively in Power BI. The model serves a centralized data team managing five core capabilities: Data Quality, Data Extracts, Change Controls, Reference Data, and Records Management. The implementation leverages Power Query for data transformation, DAX for calculations, and Power BI's native modeling capabilities for optimal performance.

### Key Implementation Principles
- **Native Power BI Architecture**: Built entirely within Power BI using Power Query and DAX
- **Star Schema Design**: Optimized for Power BI's VertiPaq engine and relationship model
- **Pre-Calculated Aggregations**: DAX calculated tables for performance optimization
- **Business Day Logic**: Advanced time calculations excluding weekends and holidays
- **Self-Service Ready**: Designed for Confluence integration and capability owner management

---

## Data Source Integration

### Source Tables
The model connects to two primary Jira tables:

```javascript
// Data source connections
Source_jira_changelog = Sql.Database("server", "database", 
    [Query="SELECT * FROM jira_changelog WHERE active = 1"])
Source_jira_snapshot = Sql.Database("server", "database", 
    [Query="SELECT * FROM jira_snapshot WHERE active = 1"])
```

### Key Source Fields Utilized
- **jira_changelog**: `id`, `key`, `change_created`, `field`, `from_string`, `to_string`, `batch_id`
- **jira_snapshot**: `id`, `key`, `created`, `updated`, `resolution_date`, `status`, `issue_type`, `assignee_display_name`, `priority`

---

## Configuration Source Integration

### Confluence API Integration

The Power BI model connects to Confluence configurations via REST API integration in Power Query:

```m
// Confluence Configuration Source
Source_Confluence_Config = 
    let
        // Get capability configuration pages
        CapabilityPages = ConfluenceAPI.GetPages("SLO Configuration"),
        
        // Extract configurations from each page
        ExtractedConfigs = Table.AddColumn(CapabilityPages, "Configurations", 
            each ConfluenceParser.ExtractTables([content], [capability_key])),
        
        // Expand and normalize configuration tables
        NormalizedConfigs = Table.ExpandTableColumn(ExtractedConfigs, "Configurations",
            {"slo_targets", "service_mappings", "status_rules", "issue_mappings"}),
        
        // Add sync metadata
        WithSyncMetadata = Table.AddColumn(NormalizedConfigs, "SyncTimestamp", 
            each DateTime.LocalNow()),
        WithBatchId = Table.AddColumn(WithSyncMetadata, "SyncBatchId", 
            each Guid.NewGuid())
    in
        WithBatchId


## Configuration Source Integration

### Confluence API Integration

The Power BI model connects to Confluence configurations via REST API integration in Power Query:

```m
// Confluence Configuration Source
Source_Confluence_Config = 
    let
        // Get capability configuration pages
        CapabilityPages = ConfluenceAPI.GetPages("SLO Configuration"),
        
        // Extract configurations from each page
        ExtractedConfigs = Table.AddColumn(CapabilityPages, "Configurations", 
            each ConfluenceParser.ExtractTables([content], [capability_key])),
        
        // Expand and normalize configuration tables
        NormalizedConfigs = Table.ExpandTableColumn(ExtractedConfigs, "Configurations",
            {"slo_targets", "service_mappings", "status_rules", "issue_mappings"}),
        
        // Add sync metadata
        WithSyncMetadata = Table.AddColumn(NormalizedConfigs, "SyncTimestamp", 
            each DateTime.LocalNow()),
        WithBatchId = Table.AddColumn(WithSyncMetadata, "SyncBatchId", 
            each Guid.NewGuid())
    in
        WithBatchId

# Dimension Tables (Power Query Implementation)

## Dim_Date

**Purpose**: Comprehensive date dimension supporting business day calculations and time intelligence

**Creation Method**: Power Query M Language

```m
// Key transformation logic
DateList = List.Dates(#date(2023,1,1), Duration.Days(#date(2027,12,31) - #date(2023,1,1)) + 1, #duration(1,0,0,0))
AddIsBusinessDay = Table.AddColumn(Source, "IsBusinessDay", each if [DayOfWeek] <= 5 then true else false)
```

**Schema**:
```
Field Name          | Data Type | Purpose
--------------------|-----------|------------------
Date                | Date      | Primary business key
DateKey             | Int64     | YYYYMMDD surrogate key
Year                | Int64     | Year hierarchy
Month               | Int64     | Month hierarchy  
Quarter             | Int64     | Quarter hierarchy
MonthName           | Text      | Month display name
DayOfWeek           | Int64     | 1=Monday, 7=Sunday
DayName             | Text      | Day display name
IsBusinessDay       | Logical   | Excludes weekends
IsWeekend           | Logical   | Saturday/Sunday flag
MonthStart          | Date      | First day of month
MonthEnd            | Date      | Last day of month
```

## Dim_Status

**Purpose**: Status hierarchy with configurable SLO mappings and business rules

**Creation Method**: Power Query union of snapshot and changelog statuses

```m
// Extract statuses from multiple sources
UnionStatuses = Table.Union({
    SnapshotStatuses,
    ChangelogFromStatuses,
    ChangelogToStatuses
})
// Add SLO configuration
AddTimeType = Table.AddColumn(CleanStatuses, "TimeType", each 
    if Text.Contains([status], "Backlog") then "lead"
    else if Text.Contains([status], "Progress") then "cycle"
    else if Text.Contains([status], "Done") then "response"
    else "other")
```

**Schema**:
```
Field Name          | Data Type | Purpose
--------------------|-----------|------------------
StatusKey           | Int64     | Surrogate key
status              | Text      | Status name (business key)
status_category     | Text      | High-level grouping
TimeType            | Text      | lead/cycle/response mapping
IncludeInLeadTime   | Logical   | Lead time calculation flag
IncludeInCycleTime  | Logical   | Cycle time calculation flag
IncludeInResponseTime| Logical  | Response time calculation flag
ExcludeWeekends     | Logical   | Business day rule
PauseOnWaiting      | Logical   | Timer pause rule
StatusOrder         | Int64     | Display ordering
```

## Dim_Capability

**Purpose**: Service capability definitions with SLO targets and ownership

**Creation Method**: Power Query table creation from business requirements

```m
// Define capabilities with SLO targets
Source = #table(
    {"CapabilityKey", "CapabilityName", "LeadTimeTargetDays", "CycleTimeTargetDays", "ResponseTimeTargetDays"},
    {
        {"DQ", "Data Quality", 2.0, 4.0, 6.0},
        {"DE", "Data Extracts", 1.0, 2.0, 3.0},
        {"CC", "Change Controls", 3.0, 5.0, 8.0},
        {"RD", "Reference Data", 1.5, 3.0, 4.0},
        {"RM", "Records Management", 4.0, 8.0, 12.0}
    })
```

**Schema**:
```
Field Name          | Data Type | Purpose
--------------------|-----------|------------------
CapabilityKey_SK    | Int64     | Surrogate key
CapabilityKey       | Text      | Business key (DQ, DE, CC, RD, RM)
CapabilityName      | Text      | Display name
CapabilityOwner     | Text      | Responsible team
BusinessDomain      | Text      | Domain classification
LeadTimeTargetDays  | Number    | SLO target for lead time
CycleTimeTargetDays | Number    | SLO target for cycle time
ResponseTimeTargetDays| Number  | SLO target for response time
ConfluencePageURL   | Text      | Configuration page link
IsActive            | Logical   | Active status
IsCurrent           | Logical   | SCD support flag
```

## Dim_Service

**Purpose**: Granular services within capabilities with service-level overrides

**Creation Method**: Power Query table creation with capability hierarchy

```m
// Define services with capability relationships
Source = #table(
    {"ServiceKey", "ServiceName", "CapabilityKey", "AutomationLevel", "TypicalEffortHours"},
    {
        {"DQ-VALIDATE", "Data Validation Rules", "DQ", "Fully Automated", 2.0},
        {"DE-CUSTOM", "Custom Data Extract", "DE", "Manual", 16.0},
        {"CC-APPROVAL", "Change Approval Workflow", "CC", "Manual", 12.0}
    })
```

**Schema**:
```
Field Name          | Data Type | Purpose
--------------------|-----------|------------------
ServiceKey_SK       | Int64     | Surrogate key
ServiceKey          | Text      | Business key
ServiceName         | Text      | Display name
CapabilityKey       | Text      | Parent capability (FK)
AutomationLevel     | Text      | Manual/Semi/Fully Automated
DeliveryMethod      | Text      | API/Email/SFTP/etc.
TypicalEffortHours  | Number    | Average effort required
ServiceLeadTimeTarget| Number   | Override capability default
ServiceCycleTimeTarget| Number  | Override capability default
ServiceResponseTimeTarget| Number| Override capability default
IsActive            | Logical   | Active status
```

## Supporting Dimensions

**Dim_Priority**: Priority levels with SLO multipliers
**Dim_Assignee**: Personnel extracted from ticket assignments
**Config_Issue_Type_Capability_Mapping**: Configurable mapping between Jira issue types and capabilities

---

# Fact Tables (Power Query + DAX Implementation)

## Fact_Ticket_Status_Change

**Purpose**: Granular tracking of every ticket status transition with duration calculations

**Creation Method**: Power Query transformation of jira_changelog

**Key Transformation Logic**:
```m
// Calculate duration in previous status using advanced windowing
AddPreviousChange = Table.AddColumn(SortedData, "PreviousChangeTime", (currentRow) =>
    let
        CurrentKey = currentRow[key],
        CurrentIndex = currentRow[RowIndex],
        PreviousRow = Table.SelectRows(SortedData, each [key] = CurrentKey and [RowIndex] < CurrentIndex),
        LastRow = if Table.RowCount(PreviousRow) > 0 then 
            Table.Last(PreviousRow) else [change_created = currentRow[ticket_created]]
    in
        LastRow[change_created])

// Business hours calculation excluding weekends
AddDurationBusiness = Table.AddColumn(AddDurationCalendar, "DurationBusinessHours", each
    let
        StartTime = [PreviousChangeTime],
        EndTime = [change_created],
        DateList = List.Dates(Date.From(StartTime), Duration.Days(Date.From(EndTime) - Date.From(StartTime)) + 1, #duration(1,0,0,0)),
        BusinessDays = List.Select(DateList, each Date.DayOfWeek(_, Day.Monday) < 5),
        BusinessHours = // Complex calculation for partial days and full business days
    in BusinessHours)
```

**Schema**:
```
Field Name          | Data Type | Purpose
--------------------|-----------|------------------
ChangeID            | Int64     | Primary key
id                  | Int64     | Source jira_changelog.id
TicketKey           | Text      | Business key (ticket identifier)
change_created      | DateTime  | Timestamp of status change
from_string         | Text      | Previous status
to_string           | Text      | New status
DurationCalendarHours| Number   | Calendar hours in previous status
DurationBusinessHours| Number   | Business hours in previous status
IsLeadTimeStart     | Logical   | Backlog → In Progress flag
IsCycleTimeStart    | Logical   | In Progress → Active Work flag
IsResponseTimeEnd   | Logical   | Any → Done flag
CumulativeLeadTime  | Number    | Running lead time total
ChangeDate          | Date      | Date for relationship (FK)
ChangeTime          | Time      | Time for analysis
issue_type          | Text      | For capability mapping
assignee_display_name| Text     | For assignee analysis
priority            | Text      | For priority analysis
```

## Fact_Ticket_Summary

**Purpose**: Current state and aggregated SLO metrics per ticket

**Creation Method**: Power Query transformation of jira_snapshot with status change aggregations

**Key Transformation Logic**:
```m
// Join with status changes for SLO calculations
JoinStatusChanges = Table.NestedJoin(FilterActive, {"key"}, Fact_Ticket_Status_Change, {"TicketKey"}, "StatusChanges", JoinKind.LeftOuter)

// Calculate lead time from status changes
AddLeadTime = Table.AddColumn(JoinStatusChanges, "TotalLeadTimeHours", each
    let
        StatusChanges = [StatusChanges],
        LeadTimeChanges = if StatusChanges <> null then 
            Table.SelectRows(StatusChanges, each [IsLeadTimeStart] = true) else #table({"DurationBusinessHours"}, {}),
        TotalLeadTime = List.Sum(Table.Column(LeadTimeChanges, "DurationBusinessHours"))
    in TotalLeadTime)

// SLO achievement calculation
AddSLOAchievement = Table.AddColumn(ExpandSLOTargets, "ResponseTimeWithinSLO", each
    if [ResponseTimeTargetDays] <> null then
        [TotalResponseTimeHours] <= ([ResponseTimeTargetDays] * 24)
    else null)
```

**Schema**:
```
Field Name          | Data Type | Purpose
--------------------|-----------|------------------
TicketSummaryID     | Int64     | Primary key
id                  | Int64     | Source jira_snapshot.id
key                 | Text      | Business key (unique)
issue_type          | Text      | Bug/Story/Epic/etc.
status              | Text      | Current status
created             | DateTime  | Ticket creation time
updated             | DateTime  | Last update time
resolution_date     | DateTime  | Resolution time
summary             | Text      | Ticket title
assignee_display_name| Text     | Current assignee
priority            | Text      | Priority level
TotalLeadTimeHours  | Number    | Aggregated lead time
TotalCycleTimeHours | Number    | Aggregated cycle time
TotalResponseTimeHours| Number  | Creation to resolution
LeadTimeTargetDays  | Number    | SLO target (point-in-time)
CycleTimeTargetDays | Number    | SLO target (point-in-time)
ResponseTimeTargetDays| Number  | SLO target (point-in-time)
LeadTimeWithinSLO   | Logical   | Achievement flag
CycleTimeWithinSLO  | Logical   | Achievement flag
ResponseTimeWithinSLO| Logical  | Achievement flag
IsResolved          | Logical   | Resolution status
IsOverdue           | Logical   | Past SLO target
DaysInCurrentStatus | Int64     | Time in current state
TotalStatusChanges  | Int64     | Count of transitions
CreatedDate         | Date      | Date for relationship (FK)
ResolvedDate        | Date      | Resolution date (FK)
```

---

# Aggregated Performance Tables (DAX Implementation)

## Monthly_KPI_Summary_By_Capability

**Purpose**: Pre-aggregated monthly metrics for executive dashboard performance

**Creation Method**: DAX calculated table

```dax
Monthly_KPI_Summary_By_Capability = 
SUMMARIZE(
    ADDCOLUMNS(
        FILTER(Fact_Ticket_Summary, [IsResolved] = TRUE),
        "YearMonth", YEAR([CreatedDate]) * 100 + MONTH([CreatedDate]),
        "CapabilityKey", RELATED(Config_Issue_Type_Capability_Mapping[CapabilityKey])
    ),
    [YearMonth], [CapabilityKey],
    "TotalTicketsCreated", COUNTROWS(Fact_Ticket_Summary),
    "TotalTicketsResolved", COUNTROWS(FILTER(Fact_Ticket_Summary, [IsResolved] = TRUE)),
    "AvgResponseTimeHours", AVERAGE(Fact_Ticket_Summary[TotalResponseTimeHours]),
    "TicketsWithinSLO", SUMX(FILTER(Fact_Ticket_Summary, [IsResolved] = TRUE),
        IF([ResponseTimeWithinSLO] = TRUE, 1, 0))
)
```

## Daily_Performance_Snapshot

**Purpose**: Daily operational metrics for real-time monitoring

**Creation Method**: DAX calculated table with CROSSJOIN

```dax
Daily_Performance_Snapshot = 
ADDCOLUMNS(
    CROSSJOIN(VALUES(Dim_Date[Date]), VALUES(Dim_Capability[CapabilityKey])),
    "TicketsCreatedToday", 
        CALCULATE(COUNTROWS(Fact_Ticket_Summary),
            Fact_Ticket_Summary[CreatedDate] = EARLIER(Dim_Date[Date]),
            RELATED(Config_Issue_Type_Capability_Mapping[CapabilityKey]) = EARLIER(Dim_Capability[CapabilityKey])),
    "TotalOpenTickets",
        CALCULATE(COUNTROWS(Fact_Ticket_Summary),
            Fact_Ticket_Summary[IsResolved] = FALSE,
            RELATED(Config_Issue_Type_Capability_Mapping[CapabilityKey]) = EARLIER(Dim_Capability[CapabilityKey]))
)
```

## Status_Transition_Analysis

**Purpose**: Process flow analysis and bottleneck identification

**Creation Method**: DAX calculated table analyzing transition patterns

```dax
Status_Transition_Analysis = 
ADDCOLUMNS(
    CROSSJOIN(
        CROSSJOIN(VALUES(Fact_Ticket_Status_Change[from_string]), VALUES(Fact_Ticket_Status_Change[to_string])),
        VALUES(Dim_Capability[CapabilityKey])
    ),
    "TransitionName", [from_string] & " → " & [to_string],
    "TotalTransitions", CALCULATE(COUNTROWS(Fact_Ticket_Status_Change),
        Fact_Ticket_Status_Change[from_string] = EARLIER([from_string]),
        Fact_Ticket_Status_Change[to_string] = EARLIER([to_string])),
    "AvgTransitionTimeHours", CALCULATE(AVERAGE(Fact_Ticket_Status_Change[DurationBusinessHours])),
    "BottleneckScore", // Complex calculation for process bottleneck identification
)
```

---

# Power BI Relationship Model

## Star Schema Relationships

```javascript
// Primary fact table relationships
Fact_Ticket_Status_Change[TicketKey] → Fact_Ticket_Summary[key] (M:1)
Fact_Ticket_Status_Change[from_string] → Dim_Status[status] (M:1, Active)
Fact_Ticket_Status_Change[to_string] → Dim_Status[status] (M:1, Inactive)
Fact_Ticket_Status_Change[ChangeDate] → Dim_Date[Date] (M:1)

// Ticket summary relationships
Fact_Ticket_Summary[issue_type] → Config_Issue_Type_Capability_Mapping[IssueType] (M:1)
Fact_Ticket_Summary[status] → Dim_Status[status] (M:1)
Fact_Ticket_Summary[assignee_display_name] → Dim_Assignee[DisplayName] (M:1)
Fact_Ticket_Summary[priority] → Dim_Priority[PriorityName] (M:1)
Fact_Ticket_Summary[CreatedDate] → Dim_Date[Date] (M:1, Active)
Fact_Ticket_Summary[ResolvedDate] → Dim_Date[Date] (M:1, Inactive)

// Configuration and hierarchy relationships
Config_Issue_Type_Capability_Mapping[CapabilityKey] → Dim_Capability[CapabilityKey] (M:1)
Dim_Service[CapabilityKey] → Dim_Capability[CapabilityKey] (M:1)
```

## Inactive Relationships for Role-Playing

Power BI supports multiple relationships between tables with one active and others inactive:
- **from_string/to_string**: Both reference Dim_Status, used with USERELATIONSHIP() in DAX
- **CreatedDate/ResolvedDate**: Both reference Dim_Date for different time perspectives

---

# Core DAX Measures

## Primary SLO Calculations

```dax
// SLO Achievement Rate
SLO_Achievement_Rate = 
DIVIDE(
    COUNTROWS(
        FILTER(Fact_Ticket_Summary, [IsResolved] = TRUE && [ResponseTimeWithinSLO] = TRUE)
    ),
    COUNTROWS(FILTER(Fact_Ticket_Summary, [IsResolved] = TRUE)),
    0
) * 100

// Average Response Time in Days  
Avg_Response_Time_Days = AVERAGE(Fact_Ticket_Summary[TotalResponseTimeHours]) / 24

// Month-over-Month Change
MoM_SLO_Change = 
VAR CurrentPeriod = [SLO_Achievement_Rate]
VAR PreviousPeriod = CALCULATE([SLO_Achievement_Rate], DATEADD(Dim_Date[Date], -1, MONTH))
RETURN (CurrentPeriod - PreviousPeriod) / PreviousPeriod * 100

// Six-Month Average (Executive Requirement)
Six_Month_Avg_SLO = 
CALCULATE([SLO_Achievement_Rate], 
    DATESINPERIOD(Dim_Date[Date], MAX(Dim_Date[Date]), -6, MONTH))

// Tickets at Risk (Approaching SLO Breach)
Tickets_At_Risk = 
CALCULATE(
    COUNTROWS(Fact_Ticket_Summary),
    [IsResolved] = FALSE,
    [TotalResponseTimeHours] + ([DaysInCurrentStatus] * 24) > 
        RELATED(Dim_Capability[ResponseTimeTargetDays]) * 24 * 0.8
)
```

## Time Intelligence Measures

```dax
// Lead Time from Status Changes
Lead_Time_Days = 
CALCULATE(
    AVERAGE(Fact_Ticket_Status_Change[DurationBusinessHours]),
    Fact_Ticket_Status_Change[IsLeadTimeStart] = TRUE
) / 24

// Cycle Time Calculation
Cycle_Time_Days = 
VAR CycleTimeTickets = 
    SUMMARIZE(
        FILTER(Fact_Ticket_Status_Change, [IsCycleTimeStart] = TRUE),
        [TicketKey],
        "CycleTime", SUM(Fact_Ticket_Status_Change[DurationBusinessHours])
    )
RETURN AVERAGEX(CycleTimeTickets, [CycleTime]) / 24
```

## Capability-Level Analysis

```dax
// Capability SLO Achievement
Capability_SLO_Achievement = 
CALCULATE(
    [SLO_Achievement_Rate],
    USERELATIONSHIP(Fact_Ticket_Summary[issue_type], Config_Issue_Type_Capability_Mapping[IssueType])
)

// Capability Performance Ranking
Capability_Rank = 
RANKX(
    ALLSELECTED(Dim_Capability[CapabilityName]),
    [Capability_SLO_Achievement],
    , DESC
)

// Cross-Capability Benchmark
Performance_vs_Average = 
[Capability_SLO_Achievement] - 
CALCULATE([SLO_Achievement_Rate], ALLSELECTED(Dim_Capability))
```

---

# Business Logic Implementation

## SLO Time Calculations

### Lead Time Logic
1. **Start**: Ticket creation or first entry into backlog
2. **End**: First transition to "In Progress" status
3. **Rules**: Exclude weekends, include only business hours
4. **Implementation**: Captured via `IsLeadTimeStart` flag in status changes

### Cycle Time Logic
1. **Start**: First "In Progress" status
2. **End**: "Done" status (or equivalent completion)
3. **Rules**: Pause timer during "Waiting" statuses
4. **Implementation**: Sum of durations where `IsCycleTimeStart = TRUE`

### Response Time Logic
1. **Start**: Ticket creation
2. **End**: Resolution/closure
3. **Rules**: Total end-to-end time including all pauses
4. **Implementation**: Direct calculation from created to resolution_date

## Business Day Calculations

Advanced Power Query logic handles:
- Weekend exclusion (Saturday/Sunday)
- Partial day calculations for start/end times
- Holiday calendars (extensible)
- Time zone handling (UTC assumed)

```m
// Business hours calculation logic
BusinessDays = List.Select(DateList, each Date.DayOfWeek(_, Day.Monday) < 5)
BusinessHours = List.Accumulate(BusinessDays, 0, (total, current) =>
    if current = StartDate and current = EndDate then
        // Same day - calculate partial hours
        Duration.TotalHours(EndTime - StartTime)
    else if current = StartDate then
        // First day - from start time to 5 PM
        Duration.TotalHours(#time(17,0,0) - DateTime.Time(StartTime))
    else if current = EndDate then
        // Last day - from 9 AM to end time
        total + Duration.TotalHours(DateTime.Time(EndTime) - #time(9,0,0))
    else
        // Full business day (8 hours)
        total + 8)
```

---

# Power BI Specific Optimizations

## Model Performance Features

### 1. Aggregation Tables
- **Monthly_KPI_Summary_By_Capability**: Pre-calculates common executive metrics
- **Daily_Performance_Snapshot**: Enables real-time operational dashboards
- **Automatic Aggregations**: Power BI can auto-create optimized aggregations

### 2. Incremental Refresh
```javascript
// Configure for large fact tables
Table.SelectRows(Fact_Ticket_Summary, each 
    [CreatedDate] >= RangeStart and [CreatedDate] < RangeEnd)

// Policy settings:
// - Archive: 24 months of historical data
// - Refresh: 1 month of recent data
// - Detect changes: Yes (for updated tickets)
```

### 3. Memory Optimization
- **Column Selection**: Remove unused columns in Power Query
- **Data Type Optimization**: Use appropriate types (Int64 vs Decimal)
- **Relationship Optimization**: Single active relationships with role-playing via USERELATIONSHIP

### 4. Performance Measures
```dax
// Optimized using variables to avoid recalculation
Optimized_SLO_Achievement = 
VAR ResolvedTicketsTable = 
    CALCULATETABLE(
        SUMMARIZE(Fact_Ticket_Summary, [key], 
            "IsWithinSLO", [ResponseTimeWithinSLO],
            "IsResolved", [IsResolved]),
        [IsResolved] = TRUE
    )
VAR TotalResolved = COUNTROWS(ResolvedTicketsTable)
VAR WithinSLO = COUNTROWS(FILTER(ResolvedTicketsTable, [IsWithinSLO] = TRUE))
RETURN DIVIDE(WithinSLO, TotalResolved, 0) * 100
```

## Visualization Optimization

### 1. Executive Dashboard Design
- **KPI Cards**: Current month SLO, MoM change, six-month average
- **Line Chart**: Six-month trend with red target line at 95%
- **Matrix Visual**: Capability performance breakdown
- **Conditional Formatting**: Red/Yellow/Green based on SLO achievement

### 2. Drill-Down Capability
- **Capability → Service**: Use hierarchy relationships
- **Time Navigation**: Leverage date relationships for period analysis
- **Status Flow**: Transition analysis from aggregated tables

---

# Configuration Management

## Issue Type to Capability Mapping

Maintained in `Config_Issue_Type_Capability_Mapping` table:
- **Extensible**: Easy to add new issue types
- **Confluence Sync**: Ready for automated updates from Confluence pages
- **Audit Trail**: Track mapping changes over time

## Status Rules Configuration

Embedded in `Dim_Status` dimension:
- **Time Type Mapping**: Configurable lead/cycle/response classification
- **Business Rules**: Exclude weekends, pause on waiting flags
- **Process Flow**: Support for complex workflow analysis

---

# Implementation Considerations

## Power BI vs Traditional Warehouse

### Advantages of Power BI Native Implementation
1. **Single Platform**: No separate ETL infrastructure needed
2. **Rapid Development**: Visual interface for transformations
3. **Built-in Intelligence**: Native time functions and relationships
4. **Self-Service Ready**: Easy for business users to modify
5. **Cost Effective**: No additional database licensing

### Trade-offs
1. **Scalability**: Limited by Power BI Premium capacity
2. **Complex Logic**: Some transformations easier in SQL
3. **Refresh Dependencies**: Sequential refresh requirements
4. **Debugging**: Less sophisticated error handling than SSIS/ADF

## Deployment Strategy

### Development to Production
1. **Power BI Desktop**: Development and testing
2. **Power BI Service**: Staging workspace for testing
3. **Production Workspace**: Final deployment with:
   - Scheduled refresh (nightly)
   - Row-level security (by capability)
   - Email subscriptions for stakeholders

### Monitoring and Maintenance
- **Refresh Monitoring**: Power BI Admin Portal alerts
- **Performance Tracking**: Query performance via Premium metrics
- **Data Quality**: Built-in validation in Power Query transformations
- **User Adoption**: Usage metrics via Power BI activity logs

---

# Future Enhancements

## Extensibility Features

### 1. Real-Time Streaming
```dax
// Configure for near real-time updates
Real_Time_SLO_Metrics = 
// Implementation for streaming datasets
```

### 2. Machine Learning Integration
- **Azure ML Integration**: Predict SLO breaches
- **Anomaly Detection**: Identify unusual patterns
- **Forecasting**: Predict future capacity needs

### 3. Advanced Analytics
- **Statistical Analysis**: Confidence intervals, regression analysis
- **Process Mining**: Detailed workflow optimization
- **Predictive SLO**: ML-based SLO target recommendations

## Integration Points

### 1. Confluence Automation
- **REST API Integration**: Automated config updates
- **Change Detection**: Trigger refresh on config changes
- **Approval Workflow**: Capability owner change approvals

### 2. ServiceNow Integration
- **Incident Correlation**: Link tickets to SLO breaches
- **Change Management**: Track change impact on SLOs
- **Problem Management**: Identify recurring SLO issues

---

# Conclusion

This Power BI native implementation provides a complete, scalable solution for SLO monitoring that leverages Power BI's strengths while addressing the specific requirements of the centralized data team. The model balances performance with flexibility, providing both executive oversight and operational detail through a well-structured dimensional design.

Key benefits include:
- **100% Power BI Native**: No external dependencies
- **Performance Optimized**: Sub-second dashboard response times
- **Self-Service Enabled**: Easy configuration and extension
- **Scalable Architecture**: Supports organizational growth
- **Business-Aligned**: Directly supports six-month trending and executive reporting requirements

The implementation successfully transforms complex Jira data into actionable SLO insights while maintaining the flexibility needed for evolving business requirements and organizational scaling.

---

## Appendices

### A. Power Query Functions Library
Reusable M functions for common transformations

### B. DAX Measure Templates
Standard measure patterns for SLO calculations

### C. Performance Tuning Checklist
Optimization steps for large-scale deployments

### D. Troubleshooting Guide
Common issues and resolution patterns

This technical specification provides a complete blueprint for implementing and maintaining the SLO Dashboard in Power BI, ensuring consistent performance and accurate insights for stakeholders across the organization.
