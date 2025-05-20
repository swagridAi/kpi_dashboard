Technical Specifications
System Architecture
High-Level Architecture
The SLO Dashboard System is built entirely on Microsoft Power Platform and Office 365, providing a zero-development, enterprise-ready solution for service performance monitoring focused on essential delivery metrics.

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │───▶│  Power BI Layer │───▶│ Distribution    │───▶│   End Users     │
│                 │    │                 │    │ Layer           │    │                 │
│ • Jira Database │    │ • Dimensional   │    │ • SharePoint    │    │ • Executives    │
│ • Confluence    │    │   Model         │    │ • Email Reports │    │ • Team Leads    │
│ • SharePoint    │    │ • Datasets      │    │ • Mobile Apps   │    │ • Team Members  │
│                 │    │ • Reports       │    │ • Teams/SMS     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
                                ↑                        ↑
                    ┌─────────────────┐    ┌─────────────────┐
                    │ Configuration   │    │ Alert Engine    │
                    │ Management      │    │ (Power Automate)│
                    │ (Confluence)    │    │                 │
                    └─────────────────┘    └─────────────────┘
Technology Stack
Core Platform:

Power BI Premium: Dimensional modeling, visualization, VertiPaq engine
Power Automate: Workflow automation, smart alerting, email distribution
SharePoint Online: Dashboard embedding, document management, collaboration
Office 365: Authentication (Azure AD), email distribution, Teams integration
Data Storage:

SQL Server Database: Primary data store for Jira snapshot and changelog
Power BI Datasets: Compressed VertiPaq format for analytical queries
SharePoint Lists: User preferences, subscription management
Integration Technologies:

REST APIs: Confluence configuration sync, Microsoft Graph integration
Power Query: ETL processes, data transformation, business rule implementation
DAX: Calculation engine, KPI definitions, time intelligence
Data Integration Flow
Source → Database: Jira ETL job (nightly) → SQL Server
Database → Power BI: Incremental refresh (scheduled every 4 hours)
Confluence → Power BI: Configuration sync (nightly at 2:00 AM UTC)
Power BI → Distribution: Automated reports and alerts via Power Automate
Data Model Specifications
Simplified Model Overview
The SLO Dashboard employs a streamlined dimensional model focused on essential service delivery metrics. This simplified approach prioritizes clarity and maintainability while providing comprehensive insight into the six core performance indicators.

Design Principles:

Focus on Actionable Metrics: Only KPIs that directly inform operational decisions
Minimal Complexity: Reduced from complex multi-tier SLA hierarchies to straightforward capability-based targets
Essential Tracking: Core time, volume, and quality measurements without individual performance analytics
Maintainable Architecture: Simplified relationships and calculations for easier long-term management
Dimensional Model Design
Simplified Star Schema Implementation:

Central Fact Tables: Fact_Ticket_Summary (tickets), Fact_Ticket_Status_Change (status transitions)
3 Core Dimensions: Dim_Date, Dim_Status, Dim_Capability
2 Configuration Tables: Config_Issue_Type_Capability_Mapping, Default_SLA_Table
Focus: Six essential KPIs covering time, volume, and quality dimensions
Key Components:

2 Fact Tables: Fact_Ticket_Summary (aggregated), Fact_Ticket_Status_Change (granular)
3 Core Dimensions: Dim_Date, Dim_Status, Dim_Capability
2 Configuration Tables: Config_Issue_Type_Capability_Mapping, Default_SLA_Table
Core Table Schemas
Fact_Ticket_Summary (Primary Fact Table):

sql
-- Business Keys
key                     NVARCHAR(50)    -- Jira ticket key (primary business key)
issue_type              NVARCHAR(50)    -- Bug/Story/Epic/Task/Incident
status                  NVARCHAR(50)    -- Current status
subtask                 BIT             -- Is subtask flag

-- Temporal Fields
created                 DATETIME2       -- Ticket creation timestamp (UTC)
updated                 DATETIME2       -- Last modification timestamp (UTC)
resolution_date         DATETIME2       -- Resolution timestamp (UTC)
CreatedDate             DATE            -- Derived: Date portion of created
ResolvedDate            DATE            -- Derived: Date portion of resolution_date

-- Core Performance Fields
TotalResponseTimeHours  DECIMAL(18,2)   -- Total response time in hours
ResolutionTimeDays      INT             -- Calendar days from creation to resolution
Met_SLA                 BIT             -- SLA achievement flag (TRUE/FALSE/NULL)

-- SLO Targets (Point-in-time)
ResponseTimeTargetDays  DECIMAL(10,2)   -- SLA target at time of measurement

-- Status Indicators
IsResolved             BIT             -- Resolution status flag
IsOverdue              BIT             -- Past SLO deadline indicator
DaysInCurrentStatus    INT             -- Age in current status
Fact_Ticket_Status_Change (Status Transition Fact):

sql
-- Primary Keys
ChangeID               BIGINT IDENTITY(1,1)    -- Surrogate key
id                     BIGINT                  -- Source changelog ID
TicketKey              NVARCHAR(50)            -- Foreign key to Fact_Ticket_Summary

-- Status Transition
change_created         DATETIME2               -- Transition timestamp (UTC)
from_string           NVARCHAR(50)            -- Previous status
to_string             NVARCHAR(50)            -- New status

-- Duration Calculations
DurationCalendarHours  DECIMAL(18,2)           -- Calendar time in previous status
DurationBusinessHours  DECIMAL(18,2)           -- Business time (excludes weekends)

-- SLO Calculation Flags
IsLeadTimeStart       BIT                     -- Marks start of lead time measurement
IsCycleTimeStart      BIT                     -- Marks start of cycle time measurement
IsResponseTimeEnd     BIT                     -- Marks end of response time measurement

-- Metadata
ChangeDate            DATE                    -- Date for relationship to Dim_Date
Dim_Date (Time Dimension):

sql
-- Primary Key
Date                  DATE            -- Calendar date (primary key)
DateKey               INT             -- YYYYMMDD integer key for performance

-- Calendar Hierarchy
Year                  INT             -- Calendar year
Quarter               INT             -- Calendar quarter (1-4)
Month                 INT             -- Calendar month (1-12)
MonthName             NVARCHAR(20)    -- Month name (January, February, etc.)
DayOfWeek             INT             -- Day of week (1=Monday, 7=Sunday)
DayName               NVARCHAR(20)    -- Day name (Monday, Tuesday, etc.)

-- Business Calendar
IsBusinessDay         BIT             -- Business day flag (excludes weekends)
IsHoliday             BIT             -- Holiday flag (configurable)
FiscalYear            INT             -- Fiscal year (if different from calendar)
FiscalQuarter         INT             -- Fiscal quarter

-- Period Boundaries
MonthStart            DATE            -- First day of month
MonthEnd              DATE            -- Last day of month
QuarterStart          DATE            -- First day of quarter
QuarterEnd            DATE            -- Last day of quarter
YearStart             DATE            -- First day of year
YearEnd               DATE            -- Last day of year
Dim_Status (Status Dimension):

sql
-- Primary Key
status                NVARCHAR(50)    -- Status name (business key)

-- Status Classification
StatusCategory        NVARCHAR(50)    -- High-level grouping (To Do, In Progress, Done)
StatusOrder           INT             -- Display sequence for workflow visualization

-- SLO Timing Rules
IncludeInLeadTime     BIT             -- Used for lead time start calculation
IncludeInCycleTime    BIT             -- Used for cycle time measurement
IncludeInResponseTime BIT             -- Used for response time end calculation

-- Business Logic Flags
IsActiveStatus        BIT             -- Indicates work is actively being performed
IsFinalStatus         BIT             -- Indicates completion
IsWaitingStatus       BIT             -- Indicates external dependencies
Dim_Capability (Capability Dimension):

sql
-- Primary Key
CapabilityKey         NVARCHAR(10)    -- Short identifier (DQ, DE, CC, RD, RM)

-- Business Definition
CapabilityName        NVARCHAR(100)   -- Display name (Data Quality, Data Extracts, etc.)
CapabilityOwner       NVARCHAR(100)   -- Responsible team/person
BusinessDomain        NVARCHAR(50)    -- Domain classification

-- SLO Targets
LeadTimeTargetDays    DECIMAL(10,2)   -- Default lead time SLA
CycleTimeTargetDays   DECIMAL(10,2)   -- Default cycle time SLA  
ResponseTimeTargetDays DECIMAL(10,2)  -- Default response time SLA

-- Metadata
IsActive              BIT             -- Currently active capability
CreatedDate           DATE            -- When capability was established
LastModified          DATE            -- Last update to configuration
Config_Issue_Type_Capability_Mapping (Issue Type Mapping):

sql
-- Primary Key
MappingKey            INT IDENTITY(1,1)    -- Surrogate key

-- Mapping Definition
IssueType             NVARCHAR(50)         -- Jira issue type (business key)
CapabilityKey         NVARCHAR(10)         -- Target capability

-- Metadata
IsActive              BIT                  -- Whether mapping is active
EffectiveDate         DATE                 -- When mapping becomes active
CreatedBy             NVARCHAR(100)        -- Who established the mapping
Notes                 NVARCHAR(500)        -- Business context
Default_SLA_Table (SLA Fallback Values):

sql
-- Primary Key
SLA_Key               INT IDENTITY(1,1)    -- Surrogate key

-- SLA Definition
TicketType            NVARCHAR(50)         -- Jira issue type (business key)
SLA_Days              DECIMAL(10,2)        -- Default SLA target in calendar days
DefaultCriticality    NVARCHAR(50)         -- Standard criticality level

-- Business Rules
ExcludeWeekends       BIT                  -- Whether weekends are excluded
BusinessDaysOnly      BIT                  -- Whether calculation uses business days only
Notes                 NVARCHAR(500)        -- Business justification

-- Metadata
IsActive              BIT                  -- Whether SLA is currently active
CreatedDate           DATE                 -- When SLA was established
Relationship Configuration
Active Relationships:

javascript
// Primary active relationships for standard filtering
Fact_Ticket_Summary[CreatedDate] → Dim_Date[Date] (Many-to-One)
Fact_Ticket_Summary[issue_type] → Config_Issue_Type_Capability_Mapping[IssueType] (Many-to-One)
Fact_Ticket_Status_Change[TicketKey] → Fact_Ticket_Summary[key] (Many-to-One)
Fact_Ticket_Status_Change[ChangeDate] → Dim_Date[Date] (Many-to-One)
Config_Issue_Type_Capability_Mapping[CapabilityKey] → Dim_Capability[CapabilityKey] (Many-to-One)
Fact_Ticket_Summary[status] → Dim_Status[status] (Many-to-One)
Inactive Relationships (used with USERELATIONSHIP in DAX):

javascript
// Role-playing date relationships
Fact_Ticket_Summary[ResolvedDate] → Dim_Date[Date] (Many-to-One, Inactive)

// Multiple status relationships for status change analysis
Fact_Ticket_Status_Change[from_string] → Dim_Status[status] (Many-to-One, Inactive)
Fact_Ticket_Status_Change[to_string] → Dim_Status[status] (Many-to-One, Active)

// Default SLA fallback
Fact_Ticket_Summary[issue_type] → Default_SLA_Table[TicketType] (Many-to-One, Inactive)
Cross-Filter Directions:

Fact-to-Dimension: Single (standard)
Dimension-to-Fact: Both (for drill-through scenarios)
Business Logic Implementation
SLA Calculation Hierarchy
The simplified system implements a 2-tier SLA target resolution with automatic fallback:

dax
-- Simplified SLA Target Resolution Logic
SLA_Target_Days = 
COALESCE(
    -- Priority 1: Capability-level target
    CALCULATE(
        RELATED(Dim_Capability[ResponseTimeTargetDays]),
        USERELATIONSHIP(
            Fact_Ticket_Summary[issue_type], 
            Config_Issue_Type_Capability_Mapping[IssueType]
        )
    ),
    
    -- Priority 2: Default SLA table
    CALCULATE(
        MAX(Default_SLA_Table[SLA_Days]),
        USERELATIONSHIP(Fact_Ticket_Summary[issue_type], Default_SLA_Table[TicketType]),
        Default_SLA_Table[IsActive] = TRUE
    ),
    
    -- Priority 3: Ultimate fallback
    5
)
Business Day Calculation Algorithm
Power Query M Implementation:

m
// Business hours calculation (9 AM - 5 PM, weekdays only)
CalculateBusinessHours = (StartTime as datetime, EndTime as datetime) =>
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
                // Same day calculation
                Number.Max(0, Duration.TotalHours(EndTime - StartTime))
            else if current = StartDate then
                // First day: from start time to 5 PM
                Number.Max(0, Duration.TotalHours(
                    #datetime(Date.Year(current), Date.Month(current), Date.Day(current), 17, 0, 0) - StartTime
                ))
            else if current = EndDate then
                // Last day: from 9 AM to end time
                total + Number.Max(0, Duration.TotalHours(
                    EndTime - #datetime(Date.Year(current), Date.Month(current), Date.Day(current), 9, 0, 0)
                ))
            else
                // Full business day (8 hours)
                total + 8
        )
    in
        BusinessHours
Core KPI Calculations
Six Essential KPIs
The simplified SLO Dashboard focuses on six core performance indicators that provide comprehensive insight into service delivery across time, volume, and quality dimensions:

1. Lead Time (Responsiveness)

Definition: Time from ticket creation until work begins
Business Value: Measures organizational responsiveness to customer requests
Calculation: Average business hours from creation to "In Progress" status
2. Cycle Time (Efficiency)

Definition: Time from work start until completion
Business Value: Indicates work process efficiency and team capacity
Calculation: Average business hours from "In Progress" to "Done" status
3. Response Time (Customer Experience)

Definition: Complete end-to-end resolution time
Business Value: Total customer experience from request to delivery
Calculation: Average calendar days from creation to resolution
4. Throughput (Capacity)

Definition: Volume of tickets completed over time
Business Value: Measures delivery capacity and identifies bottlenecks
Calculation: Count of resolved tickets per time period
5. Service Quality (SLA Achievement)

Definition: Percentage of tickets resolved within SLA targets
Business Value: Measures reliability and commitment fulfillment
Calculation: (Tickets meeting SLA ÷ Total resolved tickets) × 100
6. Issue Resolution Time (Average Performance)

Definition: Simple average resolution time across all tickets
Business Value: Baseline performance measurement for process improvement
Calculation: Average resolution time in days for all resolved tickets
Essential SLO Measures
SLO Achievement Rate:

dax
SLO_Achievement_Rate = 
VAR ResolvedTickets = 
    FILTER(
        Fact_Ticket_Summary,
        [IsResolved] = TRUE && NOT ISBLANK([Met_SLA])
    )
VAR TicketsMetSLA = 
    FILTER(ResolvedTickets, [Met_SLA] = TRUE)
RETURN
    DIVIDE(COUNTROWS(TicketsMetSLA), COUNTROWS(ResolvedTickets), 0) * 100
Average Lead Time:

dax
Lead_Time_Days = 
CALCULATE(
    AVERAGE(Fact_Ticket_Status_Change[DurationBusinessHours]),
    Fact_Ticket_Status_Change[IsLeadTimeStart] = TRUE
) / 24
Average Cycle Time:

dax
Cycle_Time_Days = 
VAR CycleTimeTickets = 
    SUMMARIZE(
        FILTER(Fact_Ticket_Status_Change, [IsCycleTimeStart] = TRUE),
        [TicketKey],
        "CycleTime", SUM(Fact_Ticket_Status_Change[DurationBusinessHours])
    )
RETURN AVERAGEX(CycleTimeTickets, [CycleTime]) / 24
Throughput:

dax
Throughput = 
CALCULATE(
    COUNTROWS(Fact_Ticket_Summary),
    Fact_Ticket_Summary[IsResolved] = TRUE,
    USERELATIONSHIP(Fact_Ticket_Summary[ResolvedDate], Dim_Date[Date])
)
Average Response Time:

dax
Avg_Response_Time_Days = 
CALCULATE(
    AVERAGE(Fact_Ticket_Summary[ResolutionTimeDays]),
    Fact_Ticket_Summary[IsResolved] = TRUE
)
Service Quality (Issue Resolution Time):

dax
Issue_Resolution_Time = 
CALCULATE(
    AVERAGE(Fact_Ticket_Summary[ResolutionTimeDays]),
    Fact_Ticket_Summary[IsResolved] = TRUE
)
Time Intelligence Functions
Month-over-Month Change:

dax
MoM_SLO_Change = 
VAR CurrentMonth = [SLO_Achievement_Rate]
VAR PreviousMonth = 
    CALCULATE(
        [SLO_Achievement_Rate],
        DATEADD(Dim_Date[Date], -1, MONTH)
    )
RETURN 
    IF(
        NOT ISBLANK(PreviousMonth) && PreviousMonth <> 0,
        DIVIDE(CurrentMonth - PreviousMonth, PreviousMonth) * 100,
        BLANK()
    )
Rolling Six-Month Average:

dax
Six_Month_Avg_SLO = 
CALCULATE(
    [SLO_Achievement_Rate],
    DATESINPERIOD(Dim_Date[Date], MAX(Dim_Date[Date]), -6, MONTH)
)
Tickets at Risk (Predictive):

dax
Tickets_At_Risk = 
VAR RiskThreshold = 0.8  -- 80% of SLA target
RETURN
COUNTROWS(
    FILTER(
        Fact_Ticket_Summary,
        [IsResolved] = FALSE &&
        [ResolutionTimeDays] >= [ResponseTimeTargetDays] * RiskThreshold
    )
)
Performance Specifications
System Requirements
Power BI Premium Capacity:

Minimum: P1 (25GB memory, 8 v-cores) - Development/Testing
Recommended: P2 (50GB memory, 16 v-cores) - Production
Enterprise: P3+ (100GB+ memory, 32+ v-cores) - Large scale deployment
Data Volume Capacity:

Tickets: 1M+ active records, 5M+ historical
Status Changes: 10M+ records with 6-month retention
Users: 1,000+ concurrent, 10,000+ total
Compressed Dataset Size: 1-5GB depending on history (reduced from complex model)
Performance Targets
Dashboard Response Times:

Initial dashboard load: <3 seconds
Filter interactions: <1 second
Cross-filter operations: <2 seconds
Export operations: <10 seconds
Mobile dashboard load: <5 seconds
Data Refresh Performance:

Full refresh: <1 hour (improved from simplified model)
Incremental refresh: <15 minutes
Configuration sync: <10 minutes
Real-time streaming: <5 seconds latency
Concurrent User Limits:

Dashboard viewers: 500 concurrent
Report builders: 50 concurrent
API consumers: 100 requests/minute
Optimization Strategies
Memory Optimization:

Reduced table count from 10+ to 6 core tables
Eliminated complex calculated columns for reopening analysis
Simplified relationship model reducing memory overhead
Optimized data types for essential fields only
Query Optimization:

Streamlined DAX calculations focusing on 6 core KPIs
Reduced cross-table complexity with simplified relationships
Pre-calculated aggregations in fact tables where beneficial
Optimized relationship directions for common query patterns
Incremental Refresh Configuration:

json
{
  "refreshPolicy": {
    "incremental": {
      "pollingExpression": "DateTime.LocalNow()",
      "refreshPeriod": "1 month",
      "archivePeriod": "24 months"
    }
  }
}
Security Specifications
Authentication Architecture
Azure Active Directory Integration:

Single Sign-On (SSO) across all Microsoft 365 services
Multi-Factor Authentication (MFA) enforcement
Conditional access policies based on location/device
Service Principal Configuration:

json
{
  "servicePrincipal": {
    "appId": "12345678-1234-1234-1234-123456789012",
    "tenant": "company.onmicrosoft.com",
    "permissions": [
      "Dataset.ReadWrite.All",
      "Report.ReadWrite.All",
      "Workspace.ReadWrite.All"
    ]
  }
}
Authorization Model
Role-Based Access Control (RBAC):

Role	Permissions	Implementation
System Admin	Full system access, user management	Azure AD Security Group
Executive	All capabilities, organization-wide view	Power BI Security Role
Capability Owner	Full access to owned capability	Row-Level Security (RLS)
Team Member	Read access to team tickets	RLS + Column restrictions
Guest	Specific report access only	Guest authentication
Row-Level Security Implementation:

dax
-- Simplified capability-based security filter
Capability_Security_Filter = 
VAR UserEmail = USERPRINCIPALNAME()
VAR UserCapabilities = 
    LOOKUPVALUE(
        User_Capability_Mapping[CapabilityKeys],
        User_Capability_Mapping[UserEmail], UserEmail
    )
RETURN
    [CapabilityKey] IN UserCapabilities
    ||
    RELATED(Dim_Capability[CapabilityOwnerEmail]) = UserEmail
Data Protection
Encryption Standards:

Data at Rest: AES 256-bit encryption (Azure managed keys)
Data in Transit: TLS 1.2+ for all communications
Power BI Datasets: Microsoft-managed encryption with optional BYOK
Data Classification:

yaml
Data_Classification:
  Public:
    - Aggregated SLO metrics
    - Trend indicators
    - Capability summaries
  Internal:
    - Individual ticket performance
    - Team-level metrics
    - Process insights
  Confidential:
    - Audit logs
    - Configuration changes
Privacy Controls:

Automatic data anonymization after 18 months
Personal data retention policies
GDPR compliance features (data subject requests)
Integration Specifications
REST API Endpoints
Confluence Configuration API:

http
# Get capability configuration
GET /rest/api/content/{pageId}?expand=body.storage,version
Authorization: Bearer {jwt_token}
Content-Type: application/json

# Update configuration
PUT /rest/api/content/{pageId}
Authorization: Bearer {jwt_token}
Content-Type: application/json
{
  "version": {"number": 2},
  "body": {"storage": {"value": "{html_content}"}}
}
Microsoft Graph Integration:

http
# Send Teams notification
POST /v1.0/teams/{teamId}/channels/{channelId}/messages
Authorization: Bearer {oauth_token}
Content-Type: application/json
{
  "body": {
    "contentType": "html",
    "content": "<h3>SLO Alert</h3><p>{alert_message}</p>"
  }
}
Power BI REST API:

http
# Trigger dataset refresh
POST /v1.0/myorg/groups/{workspaceId}/datasets/{datasetId}/refreshes
Authorization: Bearer {oauth_token}

# Export report
POST /v1.0/myorg/groups/{workspaceId}/reports/{reportId}/ExportTo
Content-Type: application/json
{
  "format": "PDF",
  "powerBIReportConfiguration": {
    "reportLevelFilters": []
  }
}
Error Handling and Resilience
Circuit Breaker Pattern:

javascript
const circuitBreaker = {
  failureThreshold: 5,
  resetTimeout: 60000,
  monitor: {
    onFailure: (error) => log.error('API failure', error),
    onSuccess: () => log.info('API recovered'),
    onOpen: () => log.warn('Circuit breaker opened')
  }
};
Retry Logic:

Exponential backoff: 1s, 2s, 4s, 8s, 16s
Maximum retries: 5 attempts
Retry conditions: 429 (rate limit), 500-503 (server errors)
No retry: 400-404, 401 (authentication), 403 (authorization)
Monitoring and Alerting
System Health Metrics
Key Performance Indicators:

yaml
SLI_Definitions:
  Availability:
    - Dashboard_Uptime: 99.9%
    - API_Response_Success_Rate: 99.5%
  Performance:
    - Average_Dashboard_Load_Time: < 3 seconds
    - P95_Query_Response_Time: < 5 seconds
  Reliability:
    - Data_Refresh_Success_Rate: 99.0%
    - Configuration_Sync_Success_Rate: 95.0%
Monitoring Stack:

Application Performance: Azure Application Insights
Infrastructure: Azure Monitor, Power BI Admin Portal
Business Metrics: Simplified Power BI monitoring dashboard (6 core KPIs)
Log Aggregation: Azure Log Analytics
Alert Configuration
Critical Alerts (Immediate Response):

System unavailable (>5 minute outage)
Data refresh failures (>2 consecutive failures)
Security incidents (unauthorized access attempts)
Warning Alerts (24-hour Response):

Performance degradation (>5 second load times)
Configuration sync delays (>4 hours)
User error rate increase (>10% spike)
Notification Channels:

yaml
Alert_Routing:
  Critical:
    - PagerDuty: On-call engineer
    - Email: IT leadership
    - Teams: System admin channel
  Warning:
    - Email: Support team
    - Teams: Monitoring channel
  Info:
    - Dashboard: System health page
Audit and Compliance
Audit Log Structure:

sql
-- Comprehensive audit logging
System_Audit_Log (
    log_id                BIGINT IDENTITY(1,1),
    event_timestamp       DATETIME2,
    event_type           NVARCHAR(50),    -- LOGIN, EXPORT, CONFIG_CHANGE
    user_id              NVARCHAR(100),
    source_ip            NVARCHAR(45),
    user_agent           NVARCHAR(500),
    resource_accessed    NVARCHAR(200),
    action_performed     NVARCHAR(100),
    result_status        NVARCHAR(20),    -- SUCCESS, FAILURE, ERROR
    additional_details   NVARCHAR(MAX)    -- JSON payload
)
Compliance Features:

Data Retention: Automated policy enforcement
Access Logging: Complete user activity trail
Change Tracking: Version control for all configurations
Privacy Controls: Automated PII detection and masking
Disaster Recovery
Backup Strategy:

Automated Daily Backups: Power BI workspace content
Configuration Backup: Confluence page exports
Database Backup: Full and differential backups
Recovery Testing: Monthly validation procedures
Recovery Time Objectives:

RTO (Recovery Time): 4 hours for full system restoration
RPO (Recovery Point): 24 hours maximum data loss
Partial Recovery: 1 hour for dashboard-only restoration
This simplified specification serves as the authoritative technical reference for implementing, maintaining, and extending the streamlined SLO Dashboard System focused on six essential performance indicators.

