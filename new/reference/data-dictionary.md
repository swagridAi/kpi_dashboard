# Data Dictionary

## Overview

This data dictionary provides comprehensive definitions for all fields, calculations, and data elements in the SLO Dashboard system. It serves as the authoritative reference for analysts, developers, and business users working with dashboard data.

---

## Source Data Fields

### Jira Snapshot Fields

The Jira snapshot table captures the current state of tickets with 44 fields. All fields and their technical specifications:

| Field Name | Data Type | Business Description |
|------------|-----------|---------------------|
| **id** | int | Unique system identifier for each ticket |
| **key** | nvarchar(255) | Business identifier (e.g., DATA-1234) |
| **created** | datetime2 | When the ticket was initially created |
| **updated** | datetime2 | Last modification timestamp |
| **resolution_date** | datetime2 | When ticket was marked as resolved/completed |
| **status** | nvarchar(255) | Current ticket status (Open, In Progress, Done, etc.) |
| **issue_type** | nvarchar(255) | Type of work (Bug, Story, Epic, Task, etc.) |
| **assignee_display_name** | nvarchar(255) | Current person responsible for the ticket |
| **priority** | nvarchar(255) | Business priority level (P1-High, P2-Medium, etc.) |
| **summary** | nvarchar(max) | Ticket title/description |
| **description** | nvarchar(max) | Detailed ticket description |
| **reporter_display_name** | nvarchar(255) | Person who created the ticket |
| **creator_display_name** | nvarchar(255) | Original ticket creator |
| **project_key** | nvarchar(255) | Project identifier |
| **project_name** | nvarchar(255) | Project display name |
| **issue_key** | nvarchar(255) | Alternative ticket key format |
| **parent_key** | nvarchar(255) | Parent ticket for sub-tasks |
| **epic_key** | nvarchar(255) | Epic this ticket belongs to |
| **epic_name** | nvarchar(255) | Epic display name |
| **sprint_id** | int | Current sprint identifier |
| **sprint_name** | nvarchar(255) | Sprint display name |
| **story_points** | float | Estimated effort points |
| **original_estimate** | bigint | Original time estimate (seconds) |
| **remaining_estimate** | bigint | Remaining time estimate (seconds) |
| **time_spent** | bigint | Actual time logged (seconds) |
| **due_date** | datetime2 | Target completion date |
| **labels** | nvarchar(max) | Comma-separated labels |
| **components** | nvarchar(max) | Ticket components |
| **fix_versions** | nvarchar(max) | Target fix versions |
| **affected_versions** | nvarchar(max) | Affected software versions |
| **environment** | nvarchar(max) | Environment information |
| **custom_field_1** | nvarchar(max) | Custom field data |
| **custom_field_2** | nvarchar(max) | Custom field data |
| **custom_field_3** | nvarchar(max) | Custom field data |
| **votes** | int | Number of votes |
| **watches** | int | Number of watchers |
| **subtask** | bit | Whether this is a sub-task (1=yes, 0=no) |
| **security_level** | nvarchar(255) | Security classification |
| **attachment_count** | int | Number of attachments |
| **comment_count** | int | Number of comments |
| **worklog_count** | int | Number of work log entries |
| **link_count** | int | Number of issue links |
| **active** | bit | Whether ticket is active (1=active, 0=archived) |
| **last_viewed** | datetime2 | Last time ticket was viewed |
| **created_timestamp** | datetime2 | Record creation timestamp |
| **updated_timestamp** | datetime2 | Record last update timestamp |

### Jira Changelog Fields  

The Jira changelog table tracks all status transitions with 16 fields:

| Field Name | Data Type | Business Description |
|------------|-----------|---------------------|
| **id** | int | Unique identifier for each status change event |
| **key** | nvarchar(255) | Ticket identifier (links to snapshot table) |
| **change_created** | datetime2 | When the status change occurred |
| **field** | nvarchar(255) | What was changed (typically "status") |
| **from_string** | nvarchar(255) | Previous status name |
| **to_string** | nvarchar(255) | New status name |
| **from_id** | nvarchar(255) | Previous status ID |
| **to_id** | nvarchar(255) | New status ID |
| **author_key** | nvarchar(255) | Who made the change |
| **author_display_name** | nvarchar(255) | Display name of change author |
| **author_email** | nvarchar(255) | Email of change author |
| **change_id** | nvarchar(255) | Change event identifier |
| **batch_id** | nvarchar(255) | Data load batch identifier |
| **active** | bit | Whether record is active (1=active, 0=archived) |
| **created_timestamp** | datetime2 | Record creation timestamp |
| **updated_timestamp** | datetime2 | Record last update timestamp |

---

## Calculated Fields

### Fact_Ticket_Summary Calculated Columns

#### ResolutionTimeDays
- **Business Definition**: Number of calendar days from ticket creation to resolution
- **Calculation Logic**: `DATEDIFF(created, COALESCE(resolution_date, NOW()), DAY)`
- **Usage Notes**: 
  - For resolved tickets: Actual resolution time
  - For open tickets: Current age of ticket
  - Always non-negative (handles data quality issues)
  - Used for SLA compliance measurement

#### Met_SLA
- **Business Definition**: Boolean indicating whether ticket was resolved within SLA target
- **Calculation Logic**: Hierarchical SLA target lookup with fallback system
  1. Service-specific SLA (if configured)
  2. Capability-level SLA (from dim_capability)
  3. Default SLA (from Default_SLA_Table)
  4. Ultimate fallback (5 days)
- **Usage Notes**:
  - `TRUE`: Ticket met SLA
  - `FALSE`: Ticket missed SLA  
  - `NULL`: Ticket not yet resolved
  - Foundation for all SLA achievement calculations

#### Was_Reopened
- **Business Definition**: Boolean indicating if ticket was ever reopened after initial resolution
- **Calculation Logic**: Detects status transitions from Done-like states back to Open-like states
- **Usage Notes**:
  - Filters out immediate corrections (< 30 minutes)
  - Requires meaningful resolution period (> 1 hour)
  - Used for service quality measurement

#### Is_Completed
- **Business Definition**: Boolean flagging tickets in completed status
- **Calculation Logic**: `TRUE` if status = "Done", "Closed", or "Resolved"
- **Usage Notes**: Pre-calculated for performance optimization in Throughput KPI

#### Reopen_Count
- **Business Definition**: Number of times ticket was reopened after initial resolution
- **Calculation Logic**: Count of status transitions from Done-like back to Open-like states
- **Usage Notes**:
  - Zero for tickets never reopened
  - Increments for each reopening event
  - Used in conjunction with Was_Reopened for detailed analysis

#### DaysInCurrentStatus
- **Business Definition**: Number of days ticket has remained in current status
- **Calculation Logic**: `Duration.Days(DateTime.LocalNow() - updated)`
- **Usage Notes**: 
  - Helps identify stagnant tickets
  - Useful for bottleneck analysis
  - Updates dynamically with each refresh

#### TotalStatusChanges
- **Business Definition**: Total number of status transitions for the ticket
- **Calculation Logic**: Count of all status changes in changelog
- **Usage Notes**:
  - Indicates workflow complexity
  - Higher counts may suggest unclear requirements
  - Used for process improvement analysis

#### CompletedDate
- **Business Definition**: Date portion of completion timestamp
- **Calculation Logic**: `Date.From(resolution_date)` if available, otherwise derived from status changes
- **Usage Notes**:
  - Used for throughput calculations
  - Enables monthly/quarterly aggregations
  - Linked to Dim_Date for time intelligence

#### SLA_Status
- **Business Definition**: Text description of SLA performance status
- **Calculation Logic**: 
  ```dax
  SWITCH(
      TRUE(),
      [resolution_date] = BLANK() AND [ResolutionTimeDays] > [SLA_Target], "Open - SLA Breached",
      [resolution_date] = BLANK(), "Open - Within SLA", 
      [Met_SLA] = TRUE, "Resolved - SLA Met",
      [Met_SLA] = FALSE, "Resolved - SLA Missed",
      "Unknown"
  )
  ```
- **Usage Notes**: Human-readable status for reporting and alerts

### Key Measures

#### SLO_Achievement_Rate
- **Business Definition**: Percentage of resolved tickets that met their SLA target
- **DAX Formula**: 
  ```dax
  DIVIDE(
      COUNTROWS(FILTER(Fact_Ticket_Summary, [Met_SLA] = TRUE)),
      COUNTROWS(FILTER(Fact_Ticket_Summary, [Met_SLA] <> BLANK())),
      0
  ) * 100
  ```
- **Interpretation**: Higher percentages indicate better SLA performance

#### Throughput
- **Business Definition**: Count of completed tickets within selected time period
- **Calculation Method**: 
  ```dax
  CALCULATE(
      COUNTROWS(Fact_Ticket_Summary),
      Fact_Ticket_Summary[Is_Completed] = TRUE,
      USERELATIONSHIP(Fact_Ticket_Summary[CompletedDate], Dim_Date[Date])
  )
  ```
- **Filtering**: Automatically respects date and capability filters

#### Lead_Time_Days
- **Business Definition**: Average time from ticket creation/backlog entry until work begins
- **DAX Formula**:
  ```dax
  CALCULATE(
      AVERAGE(Fact_Ticket_Status_Change[DurationBusinessHours]),
      Fact_Ticket_Status_Change[IsLeadTimeStart] = TRUE
  ) / 24
  ```
- **Usage Notes**: Measures organizational responsiveness to new requests

#### Cycle_Time_Days
- **Business Definition**: Average time from work start until completion
- **DAX Formula**:
  ```dax
  VAR CycleTimeTickets = 
      SUMMARIZE(
          FILTER(Fact_Ticket_Status_Change, [IsCycleTimeStart] = TRUE),
          [TicketKey],
          "CycleTime", SUM(Fact_Ticket_Status_Change[DurationBusinessHours])
      )
  RETURN AVERAGEX(CycleTimeTickets, [CycleTime]) / 24
  ```
- **Usage Notes**: Indicates work process efficiency

#### Response_Time_Days
- **Business Definition**: Average total time from ticket creation to final resolution
- **Calculation Method**: `AVERAGE(Fact_Ticket_Summary[ResolutionTimeDays])` for resolved tickets
- **Usage Notes**: End-to-end customer experience measurement, includes all delays

#### Tickets_At_Risk
- **Business Definition**: Count of open tickets approaching SLA breach
- **DAX Formula**:
  ```dax
  CALCULATE(
      COUNTROWS(Fact_Ticket_Summary),
      [resolution_date] = BLANK(),
      [ResolutionTimeDays] >= [SLA_Target_Days] * 0.8
  )
  ```
- **Usage Notes**: Early warning indicator for proactive management

#### MoM_SLO_Change
- **Business Definition**: Month-over-month change in SLO achievement rate
- **DAX Formula**:
  ```dax
  VAR CurrentPeriod = [SLO_Achievement_Rate]
  VAR PreviousPeriod = CALCULATE([SLO_Achievement_Rate], DATEADD(Dim_Date[Date], -1, MONTH))
  RETURN (CurrentPeriod - PreviousPeriod)
  ```
- **Usage Notes**: Trend analysis for performance improvement tracking

#### Six_Month_Avg_SLO
- **Business Definition**: Rolling six-month average SLO achievement
- **DAX Formula**:
  ```dax
  CALCULATE(
      [SLO_Achievement_Rate], 
      DATESINPERIOD(Dim_Date[Date], MAX(Dim_Date[Date]), -6, MONTH)
  )
  ```
- **Usage Notes**: Smooths monthly variations for strategic planning

#### First_Pass_Resolution_Rate
- **Business Definition**: Percentage of tickets resolved correctly on first attempt
- **DAX Formula**:
  ```dax
  DIVIDE(
      COUNTROWS(FILTER(Fact_Ticket_Summary, [IsResolved] = TRUE && [Was_Reopened] = FALSE)),
      COUNTROWS(FILTER(Fact_Ticket_Summary, [IsResolved] = TRUE)),
      0
  ) * 100
  ```
- **Usage Notes**: Service quality indicator, complements speed metrics

---

## Dimension Attributes

### Dim_Date
- **Date**: Primary date field for relationships
- **Year, Month, Quarter**: Hierarchical time groupings
- **IsBusinessDay**: Boolean excluding weekends
- **MonthStart, MonthEnd**: Month boundary calculations for period logic
- **Business Calendar**: Supports holiday exclusions and business day calculations

### Dim_Status
- **status**: Status name (business key)
- **status_category**: High-level grouping (To Do, In Progress, Done)
- **TimeType**: SLO classification (lead, cycle, response, other)
- **IncludeInLeadTime**: Boolean for lead time start calculation
- **IncludeInCycleTime**: Boolean for cycle time calculation  
- **IncludeInResponseTime**: Boolean for response time end calculation
- **StatusOrder**: Display sequence for workflow visualization

### Dim_Capability
- **CapabilityKey**: Short identifier (DQ, DE, CC, RD, RM)
- **CapabilityName**: Display name (Data Quality, Data Extracts, etc.)
- **LeadTimeTargetDays**: Default lead time SLA
- **CycleTimeTargetDays**: Default cycle time SLA
- **ResponseTimeTargetDays**: Default response time SLA
- **CapabilityOwner**: Responsible team/person
- **BusinessDomain**: Domain classification

### Default_SLA_Table
- **SLA_Key**: Surrogate key
- **TicketType**: Jira issue type (business key)
- **SLA_Days**: Default SLA target in calendar days
- **SLA_Type**: Type of SLA (Response Time, Lead Time, Cycle Time)
- **DefaultCriticality**: Standard criticality level (High, Medium, Low)
- **ExcludeWeekends**: Whether weekends are excluded from calculation
- **BusinessDaysOnly**: Whether calculation uses business days only
- **Notes**: Business justification for SLA target
- **IsActive**: Whether this SLA is currently active
- **CreatedDate**: When SLA record was created
- **CreatedBy**: Who established the SLA
- **LastModified**: Last modification timestamp

### Dim_Service
- **ServiceKey**: Unique service identifier (e.g., "DQ-VALIDATE", "DE-CUSTOM")
- **ServiceName**: Display name (e.g., "Data Validation Rules", "Custom Data Extract")
- **CapabilityKey**: Parent capability foreign key 
- **AutomationLevel**: Process automation (Manual, Semi-Automated, Fully Automated)
- **DeliveryMethod**: How service is delivered (API, Email, SFTP, Portal)
- **TypicalEffortHours**: Average effort required per request
- **ServiceLeadTimeTarget**: Service-specific lead time override
- **ServiceCycleTimeTarget**: Service-specific cycle time override
- **ServiceResponseTimeTarget**: Service-specific response time override
- **IsActive**: Whether service is currently offered

### Dim_Priority
- **PriorityKey**: Surrogate key
- **PriorityName**: Display name (Critical, High, Medium, Low)
- **PriorityLevel**: Numeric ordering (1=highest, 4=lowest)
- **SLAMultiplier**: Factor applied to SLA targets (0.5=50% faster for high priority)
- **EscalationHours**: Hours before escalation required
- **BusinessImpact**: Description of business impact level
- **ResponseExpectation**: Expected response time description

### Dim_Assignee  
- **AssigneeKey**: Surrogate key
- **DisplayName**: Person's display name
- **EmailAddress**: Contact email
- **UserKey**: Jira user identifier
- **CapabilityKey**: Primary capability assignment
- **TeamLead**: Boolean indicating team leadership role
- **IsActive**: Whether person is currently active
- **ManagerName**: Direct manager
- **Department**: Organizational department

### Config_Issue_Type_Capability_Mapping
- **MappingKey**: Surrogate key
- **IssueType**: Jira issue type (Bug, Story, Epic, etc.)
- **CapabilityKey**: Target capability
- **ServiceKey**: Specific service (optional)
- **IsDefault**: Whether this is the default mapping
- **EffectiveDate**: When mapping becomes active
- **ExpirationDate**: When mapping expires (null = permanent)
- **CreatedBy**: Who established the mapping
- **ApprovedBy**: Who approved the mapping

---

## Data Quality Notes

### Known Data Issues

**Missing Resolution Dates**
- **Issue**: Some resolved tickets lack resolution_date
- **Workaround**: Use timestamp of last status change to Done/Closed/Resolved
- **Impact**: May affect resolution time accuracy by hours/days

**Inconsistent Status Names**  
- **Issue**: Historical status changes may use different naming conventions
- **Workaround**: Status mapping tables standardize variations
- **Impact**: Minimal after mapping application

**Timezone Handling**
- **Issue**: Source timestamps may be in different timezones
- **Workaround**: All calculations convert to UTC, Dim_Date handles local business calendar
- **Impact**: Business day calculations reflect local business hours

**Incomplete Status Workflows**
- **Issue**: Some capabilities may have status names not mapped in Dim_Status
- **Workaround**: Default time type categories applied with manual review process
- **Impact**: May require periodic review and updates to status mappings

### Validation Rules

**Data Integrity Checks**
- Resolution dates must not precede creation dates
- Status transitions must follow logical workflow progression  
- SLA targets must be positive numbers
- Active tickets must have valid status values
- Completed tickets should have resolution dates within 48 hours of status change

**Business Rule Validation**
- Was_Reopened must be consistent with Reopen_Count (Was_Reopened = TRUE ↔ Reopen_Count > 0)
- Met_SLA must align with resolution time vs. SLA target comparison
- Is_Completed must match current status categorization (Done, Closed, Resolved)
- DaysInCurrentStatus must be non-negative and logical
- TotalStatusChanges must equal actual count in changelog table

**Cross-Table Validation**
- All tickets in Fact_Ticket_Summary must exist in source Jira snapshot
- All status changes in Fact_Ticket_Status_Change must link to valid tickets
- Issue types in fact tables must have mappings in Config_Issue_Type_Capability_Mapping
- All capabilities referenced must exist in Dim_Capability
- Date relationships must have corresponding records in Dim_Date

---

## Field Relationships

### Primary Relationships
- `Fact_Ticket_Summary[key] ↔ Fact_Ticket_Status_Change[TicketKey]` (1:Many)
- `Fact_Ticket_Summary[CreatedDate] ↔ Dim_Date[Date]` (Many:1, Active)
- `Fact_Ticket_Summary[ResolvedDate] ↔ Dim_Date[Date]` (Many:1, Inactive)

### Lookup Relationships
- `Fact_Ticket_Summary[issue_type] ↔ Config_Issue_Type_Capability_Mapping[IssueType]` (Many:1)
- `Config_Issue_Type_Capability_Mapping[CapabilityKey] ↔ Dim_Capability[CapabilityKey]` (Many:1)
- `Fact_Ticket_Summary[issue_type] ↔ Default_SLA_Table[TicketType]` (Many:1, Inactive)
- `Fact_Ticket_Summary[status] ↔ Dim_Status[status]` (Many:1)
- `Fact_Ticket_Summary[assignee_display_name] ↔ Dim_Assignee[DisplayName]` (Many:1)
- `Fact_Ticket_Summary[priority] ↔ Dim_Priority[PriorityName]` (Many:1)
- `Fact_Ticket_Status_Change[from_string] ↔ Dim_Status[status]` (Many:1, Active)
- `Fact_Ticket_Status_Change[to_string] ↔ Dim_Status[status]` (Many:1, Inactive)

### Cross-Filter Behavior
- Most relationships use "Both" directions for flexible analysis
- Date relationships support time intelligence functions
- Inactive relationships activated via USERELATIONSHIP() in DAX measures
- Row-level security applied at capability level through issue type mapping

---

## Usage Guidelines

### Performance Optimization

**Pre-calculated boolean fields** (Is_Completed, Met_SLA, Was_Reopened) improve query performance by avoiding runtime string comparisons and complex logic evaluation.

**Boolean and date filters** are pushed to the storage engine, leveraging Power BI's VertiPaq compression and query optimization.

**Time intelligence measures** should be used instead of calculated columns for period-based calculations to maintain optimal model performance.

**Active vs. inactive** ticket filtering should be applied consistently using `active = 1` filter unless historical analysis specifically requires archived records.

### Data Interpretation Guidelines

**Reopened tickets** indicate potential quality issues requiring root cause investigation. High reopening rates may suggest:
- Unclear requirements gathering
- Insufficient testing before resolution
- Process gaps in the workflow
- Training opportunities for resolvers

**Throughput trends** help identify capacity constraints and seasonal patterns. Declining throughput may indicate:
- Resource constraints
- Process bottlenecks
- Increasing work complexity
- System or tooling issues

**SLA variance analysis** reveals process improvement opportunities. Consistent SLA misses may indicate:
- Unrealistic target setting
- Process inefficiencies
- Resource allocation issues
- External dependencies impact

### Time-Based Analysis Patterns

**Business Day SLA Calculations**
```dax
Business_Days_SLA_Achievement = 
CALCULATE(
    [SLO_Achievement_Rate],
    FILTER(
        Fact_Ticket_Summary,
        [ResolutionTimeBusinessDays] <= RELATED(Default_SLA_Table[SLA_Days])
    )
)
```

**Capability Performance Comparison**
```dax
CALCULATE(
    [Throughput],
    RELATED(Config_Issue_Type_Capability_Mapping[CapabilityKey]) = "DQ"
)
```

**Rolling Period Calculations**
```dax
Rolling_3_Month_SLO = 
CALCULATE(
    [SLO_Achievement_Rate],
    DATESINPERIOD(Dim_Date[Date], MAX(Dim_Date[Date]), -3, MONTH)
)
```

**Service Quality Analysis**
```dax
Quality_Trend = 
CALCULATE(
    [First_Pass_Resolution_Rate],
    DATEADD(Dim_Date[Date], -1, MONTH)
) 
```

**Risk Assessment**
```dax
High_Risk_Tickets = 
CALCULATE(
    COUNTROWS(Fact_Ticket_Summary),
    AND(
        [resolution_date] = BLANK(),
        [ResolutionTimeDays] > [SLA_Target_Days] * 0.9
    )
)
```

---

This data dictionary serves as the foundation for accurate, consistent analysis across the SLO Dashboard system. Regular updates ensure alignment with evolving business requirements and system enhancements.