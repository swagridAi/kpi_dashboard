Data Dictionary
Overview
This data dictionary provides comprehensive definitions for all fields, calculations, and data elements in the simplified SLO Dashboard system. The system focuses on 6 core KPIs that provide essential insights into service delivery performance: Lead Time, Cycle Time, Response Time, Throughput, Service Quality, and Issue Resolution.

Source Data Fields
Jira Snapshot Fields
The Jira snapshot table captures the current state of tickets with 44 fields. All fields and their technical specifications:
Field NameData TypeBusiness DescriptionidintUnique system identifier for each ticketkeynvarchar(255)Business identifier (e.g., DATA-1234)createddatetime2When the ticket was initially createdupdateddatetime2Last modification timestampresolution_datedatetime2When ticket was marked as resolved/completedstatusnvarchar(255)Current ticket status (Open, In Progress, Done, etc.)issue_typenvarchar(255)Type of work (Bug, Story, Epic, Task, etc.)assignee_display_namenvarchar(255)Current person responsible for the ticketprioritynvarchar(255)Business priority level (P1-High, P2-Medium, etc.)summarynvarchar(max)Ticket title/descriptiondescriptionnvarchar(max)Detailed ticket descriptionreporter_display_namenvarchar(255)Person who created the ticketcreator_display_namenvarchar(255)Original ticket creatorproject_keynvarchar(255)Project identifierproject_namenvarchar(255)Project display nameissue_keynvarchar(255)Alternative ticket key formatparent_keynvarchar(255)Parent ticket for sub-tasksepic_keynvarchar(255)Epic this ticket belongs toepic_namenvarchar(255)Epic display namesprint_idintCurrent sprint identifiersprint_namenvarchar(255)Sprint display namestory_pointsfloatEstimated effort pointsoriginal_estimatebigintOriginal time estimate (seconds)remaining_estimatebigintRemaining time estimate (seconds)time_spentbigintActual time logged (seconds)due_datedatetime2Target completion datelabelsnvarchar(max)Comma-separated labelscomponentsnvarchar(max)Ticket componentsfix_versionsnvarchar(max)Target fix versionsaffected_versionsnvarchar(max)Affected software versionsenvironmentnvarchar(max)Environment informationcustom_field_1nvarchar(max)Custom field datacustom_field_2nvarchar(max)Custom field datacustom_field_3nvarchar(max)Custom field datavotesintNumber of voteswatchesintNumber of watcherssubtaskbitWhether this is a sub-task (1=yes, 0=no)security_levelnvarchar(255)Security classificationattachment_countintNumber of attachmentscomment_countintNumber of commentsworklog_countintNumber of work log entrieslink_countintNumber of issue linksactivebitWhether ticket is active (1=active, 0=archived)last_vieweddatetime2Last time ticket was viewedcreated_timestampdatetime2Record creation timestampupdated_timestampdatetime2Record last update timestamp
Jira Changelog Fields
The Jira changelog table tracks all status transitions with 16 fields:
Field NameData TypeBusiness DescriptionidintUnique identifier for each status change eventkeynvarchar(255)Ticket identifier (links to snapshot table)change_createddatetime2When the status change occurredfieldnvarchar(255)What was changed (typically "status")from_stringnvarchar(255)Previous status nameto_stringnvarchar(255)New status namefrom_idnvarchar(255)Previous status IDto_idnvarchar(255)New status IDauthor_keynvarchar(255)Who made the changeauthor_display_namenvarchar(255)Display name of change authorauthor_emailnvarchar(255)Email of change authorchange_idnvarchar(255)Change event identifierbatch_idnvarchar(255)Data load batch identifieractivebitWhether record is active (1=active, 0=archived)created_timestampdatetime2Record creation timestampupdated_timestampdatetime2Record last update timestamp

Calculated Fields
Fact_Ticket_Summary Calculated Columns
ResolutionTimeDays

Business Definition: Number of calendar days from ticket creation to resolution
Calculation Logic: DATEDIFF(created, COALESCE(resolution_date, NOW()), DAY)
Usage Notes:

For resolved tickets: Actual resolution time
For open tickets: Current age of ticket
Always non-negative (handles data quality issues)
Foundation for all time-based KPIs (Lead Time, Cycle Time, Response Time)



Met_SLA

Business Definition: Boolean indicating whether ticket was resolved within SLA target
Calculation Logic: Simplified 2-tier SLA target resolution:

Capability-level SLA (from Dim_Capability)
Default SLA fallback (5 days)


Usage Notes:

TRUE: Ticket met SLA
FALSE: Ticket missed SLA
NULL: Ticket not yet resolved
Foundation for Service Quality KPI calculation



Is_Completed

Business Definition: Boolean flagging tickets in completed status
Calculation Logic: TRUE if status = "Done", "Closed", or "Resolved"
Usage Notes: Pre-calculated for performance optimization in Throughput KPI

DaysInCurrentStatus

Business Definition: Number of days ticket has remained in current status
Calculation Logic: Duration.Days(DateTime.LocalNow() - updated)
Usage Notes:

Helps identify stagnant tickets
Useful for bottleneck analysis
Updates dynamically with each refresh



CompletedDate

Business Definition: Date portion of completion timestamp
Calculation Logic: Date.From(resolution_date) if available, otherwise derived from status changes
Usage Notes:

Used for Throughput calculations
Enables monthly/quarterly aggregations
Linked to Dim_Date for time intelligence




6 Core KPI Measures
Time-Based KPIs
Lead_Time_Days

Business Definition: Average time from ticket creation/backlog entry until work begins
DAX Formula:
daxCALCULATE(
    AVERAGE(Fact_Ticket_Status_Change[DurationBusinessHours]),
    Fact_Ticket_Status_Change[IsLeadTimeStart] = TRUE
) / 24

Usage Notes: Measures organizational responsiveness to new requests

Cycle_Time_Days

Business Definition: Average time from work start until completion
DAX Formula:
daxVAR CycleTimeTickets = 
    SUMMARIZE(
        FILTER(Fact_Ticket_Status_Change, [IsCycleTimeStart] = TRUE),
        [TicketKey],
        "CycleTime", SUM(Fact_Ticket_Status_Change[DurationBusinessHours])
    )
RETURN AVERAGEX(CycleTimeTickets, [CycleTime]) / 24

Usage Notes: Indicates work process efficiency

Response_Time_Days (Issue Resolution)

Business Definition: Average total time from ticket creation to final resolution
Calculation Method: AVERAGE(Fact_Ticket_Summary[ResolutionTimeDays]) for resolved tickets
Usage Notes: End-to-end customer experience measurement, includes all delays

Volume KPI
Throughput

Business Definition: Count of completed tickets within selected time period
Calculation Method:
daxCALCULATE(
    COUNTROWS(Fact_Ticket_Summary),
    Fact_Ticket_Summary[Is_Completed] = TRUE,
    USERELATIONSHIP(Fact_Ticket_Summary[CompletedDate], Dim_Date[Date])
)

Filtering: Automatically respects date and capability filters

Quality KPI
SLO_Achievement_Rate (Service Quality)

Business Definition: Percentage of resolved tickets that met their SLA target
DAX Formula:
daxDIVIDE(
    COUNTROWS(FILTER(Fact_Ticket_Summary, [Met_SLA] = TRUE)),
    COUNTROWS(FILTER(Fact_Ticket_Summary, [Met_SLA] <> BLANK())),
    0
) * 100

Interpretation: Higher percentages indicate better SLA performance


Dimension Attributes
Dim_Date

Date: Primary date field for relationships
Year, Month, Quarter: Hierarchical time groupings
IsBusinessDay: Boolean excluding weekends
MonthStart, MonthEnd: Month boundary calculations for period logic
Business Calendar: Supports holiday exclusions and business day calculations

Dim_Status

status: Status name (business key)
status_category: High-level grouping (To Do, In Progress, Done)
TimeType: SLO classification (lead, cycle, response, other)
IncludeInLeadTime: Boolean for lead time start calculation
IncludeInCycleTime: Boolean for cycle time calculation
IncludeInResponseTime: Boolean for response time end calculation
StatusOrder: Display sequence for workflow visualization

Dim_Capability

CapabilityKey: Short identifier (DQ, DE, CC, RD, RM)
CapabilityName: Display name (Data Quality, Data Extracts, etc.)
LeadTimeTargetDays: Default lead time SLA
CycleTimeTargetDays: Default cycle time SLA
ResponseTimeTargetDays: Default response time SLA
CapabilityOwner: Responsible team/person

Default_SLA_Table

SLA_Key: Surrogate key
TicketType: Jira issue type (business key)
SLA_Days: Default SLA target in calendar days
Notes: Business justification for SLA target
IsActive: Whether this SLA is currently active
CreatedDate: When SLA record was created

Config_Issue_Type_Capability_Mapping

MappingKey: Surrogate key
IssueType: Jira issue type (Bug, Story, Epic, etc.)
CapabilityKey: Target capability
Notes: Business context for mapping
IsActive: Whether mapping is currently active
EffectiveDate: When mapping becomes active


Data Quality Notes
Known Data Issues
Missing Resolution Dates

Issue: Some resolved tickets lack resolution_date
Workaround: Use timestamp of last status change to Done/Closed/Resolved
Impact: May affect resolution time accuracy by hours/days

Inconsistent Status Names

Issue: Historical status changes may use different naming conventions
Workaround: Status mapping tables standardize variations
Impact: Minimal after mapping application

Timezone Handling

Issue: Source timestamps may be in different timezones
Workaround: All calculations convert to UTC, Dim_Date handles local business calendar
Impact: Business day calculations reflect local business hours

Validation Rules
Data Integrity Checks

Resolution dates must not precede creation dates
Status transitions must follow logical workflow progression
SLA targets must be positive numbers
Active tickets must have valid status values
Completed tickets should have resolution dates within 48 hours of status change

Business Rule Validation

Met_SLA must align with resolution time vs. SLA target comparison
Is_Completed must match current status categorization (Done, Closed, Resolved)
DaysInCurrentStatus must be non-negative and logical

Cross-Table Validation

All tickets in Fact_Ticket_Summary must exist in source Jira snapshot
All status changes in Fact_Ticket_Status_Change must link to valid tickets
Issue types in fact tables must have mappings in Config_Issue_Type_Capability_Mapping
All capabilities referenced must exist in Dim_Capability
Date relationships must have corresponding records in Dim_Date


Field Relationships
Primary Relationships

Fact_Ticket_Summary[key] ↔ Fact_Ticket_Status_Change[TicketKey] (1:Many)
Fact_Ticket_Summary[CreatedDate] ↔ Dim_Date[Date] (Many:1, Active)
Fact_Ticket_Summary[CompletedDate] ↔ Dim_Date[Date] (Many:1, Inactive)

Lookup Relationships

Fact_Ticket_Summary[issue_type] ↔ Config_Issue_Type_Capability_Mapping[IssueType] (Many:1)
Config_Issue_Type_Capability_Mapping[CapabilityKey] ↔ Dim_Capability[CapabilityKey] (Many:1)
Fact_Ticket_Summary[issue_type] ↔ Default_SLA_Table[TicketType] (Many:1, Inactive)
Fact_Ticket_Summary[status] ↔ Dim_Status[status] (Many:1)
Fact_Ticket_Status_Change[from_string] ↔ Dim_Status[status] (Many:1, Active)
Fact_Ticket_Status_Change[to_string] ↔ Dim_Status[status] (Many:1, Inactive)

Cross-Filter Behavior

Most relationships use "Both" directions for flexible analysis
Date relationships support time intelligence functions
Inactive relationships activated via USERELATIONSHIP() in DAX measures
Row-level security applied at capability level through issue type mapping


Simplified SLA Hierarchy
The system uses a streamlined 2-tier SLA target resolution:
daxSLA_Target_Days = 
COALESCE(
    -- Priority 1: Capability-level SLA target
    RELATED(Dim_Capability[ResponseTimeTargetDays]),
    
    -- Priority 2: Default fallback
    5
)
This simplified approach ensures every ticket has an SLA target while reducing system complexity. Capability owners can set specific targets for their teams, with a 5-day fallback ensuring no tickets are left without SLA measurement.

Usage Guidelines
Performance Optimization
Pre-calculated boolean fields (Is_Completed, Met_SLA) improve query performance by avoiding runtime string comparisons and complex logic evaluation.
Boolean and date filters are pushed to the storage engine, leveraging Power BI's VertiPaq compression and query optimization.
Time intelligence measures should be used instead of calculated columns for period-based calculations to maintain optimal model performance.
Active vs. inactive ticket filtering should be applied consistently using active = 1 filter unless historical analysis specifically requires archived records.
Data Interpretation Guidelines
Time-based KPIs help identify process bottlenecks:

Lead Time issues indicate capacity constraints or prioritization problems
Cycle Time problems suggest workflow inefficiencies or complexity issues
Response Time represents the complete customer experience

Throughput trends help identify capacity constraints and seasonal patterns. Declining throughput may indicate:

Resource constraints
Process bottlenecks
Increasing work complexity
System or tooling issues

Service Quality (SLO achievement) reveals process improvement opportunities. Consistent SLA misses may indicate:

Unrealistic target setting
Process inefficiencies
Resource allocation issues
External dependencies impact

Core KPI Analysis Patterns
Time-Based Analysis
dax-- Capability performance comparison
CALCULATE(
    [Response_Time_Days],
    RELATED(Config_Issue_Type_Capability_Mapping[CapabilityKey]) = "DQ"
)
Volume Analysis
dax-- Monthly throughput by capability
CALCULATE(
    [Throughput],
    DATESMTD(Dim_Date[Date])
)
Quality Assessment
dax-- Service quality by issue type
CALCULATE(
    [SLO_Achievement_Rate],
    Fact_Ticket_Summary[issue_type] = "Bug"
)

This simplified data dictionary focuses on the essential elements needed to understand and work with the 6 core KPIs while maintaining data accuracy and system performance. Regular updates ensure alignment with evolving business requirements and system enhancements.