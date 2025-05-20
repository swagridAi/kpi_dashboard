SLA System Customization
Overview
The SLA system provides a streamlined, two-tier approach to service level management that balances organizational consistency with capability-specific requirements. This guide covers essential configuration options for managing SLA targets that directly support the six core KPIs: Lead Time, Cycle Time, Response Time, Throughput, Service Quality, and Issue Resolution Time.
What This System Does NOT Include
To maintain simplicity and focus, this system explicitly excludes:

Service-specific SLA overrides within capabilities
Priority-based SLA adjustments (P1, P2, P3, P4)
Individual assignee performance tracking
Complex reopening pattern analysis
Multi-tier approval workflows for SLA changes

SLA Framework Architecture
Two-Tier SLA Resolution
The system employs a simplified two-tier fallback mechanism for SLA target resolution, ensuring every ticket has an appropriate SLA target:

Capability-Level SLA Targets (primary)

Team-defined SLAs configured by capability owners
Based on business commitments and operational capacity


Default SLA Table Values (fallback)

Organizational fallback SLAs for all ticket types
Ensures coverage when capability-specific targets are unavailable



Technical Implementation
dax// Complete SLA target resolution with simplified hierarchy
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
How SLA Customization Supports Core KPIs
Direct KPI Support:

Lead Time, Cycle Time, Response Time: SLA targets define measurement boundaries and success criteria
Service Quality: SLA achievement calculation forms the foundation of quality measurement
Issue Resolution Time: SLA targets provide baseline expectations for resolution timing

Indirect KPI Support:

Throughput: Realistic SLA targets create appropriate pressure for consistent delivery without sacrificing quality

Default SLA Configuration
Default SLA Table Structure
The Default SLA Table serves as the foundation for SLA management across the organization:
Field NameData TypePurposeTicketTypeTextJira issue type (business key)SLA_DaysNumberDefault SLA target in business daysExcludeWeekendsBooleanWeekend exclusion flagBusinessDaysOnlyBooleanBusiness days calculation flagNotesTextBusiness justification
Standard Default Values
Ticket TypeSLA DaysBusiness JustificationBug3Critical defects require faster responseTask5Standard business day response targetStory7User stories standard processingEpic10Large initiatives allow longer response timeSub-task2Quick components support main workIncident1Production incidents highest priorityService Request5Standard service deliveryChange Request10Change management processImprovement7Enhancement requestsNew Feature15Extended analysis time required
Data Source Configuration
SharePoint List (Recommended for Production)
m// Collaborative editing with version control
let
    Source = SharePoint.Tables("https://company.sharepoint.com/sites/DataTeam"),
    SLA_List = Source{[Name="Default_SLA_Configuration"]}[Items],
    
    // Handle SharePoint column formatting
    RenameColumns = Table.RenameColumns(SLA_List, {
        {"Title", "TicketType"},
        {"SLA_x0020_Days", "SLA_Days"}
    }),
    
    // Data validation and type conversion
    FilterActive = Table.SelectRows(RenameColumns, each [IsActive] = true),
    ValidateData = Table.TransformColumns(FilterActive, {
        {"SLA_Days", each try Number.From(_) otherwise 5},
        {"ExcludeWeekends", each try Logical.From(_) otherwise true}
    })
in
    ValidateData
Capability-Level Customization
Configuration Authority

Capability owners have approval authority for their team's SLAs
Changes require business justification
Significant changes (>25% adjustment) require executive approval

Configuration Process

Baseline Analysis

Review 6-month historical performance
Identify capability-specific factors
Consider workload characteristics and resource constraints


Target Setting

Set realistic, achievable targets (aim for 90-95% achievement rate)
Align with customer expectations and business commitments
Balance stretch goals with operational capability


Implementation

Update Dim_Capability[ResponseTimeTargetDays] in configuration
Communicate changes to stakeholders
Monitor post-implementation performance


Validation

Track SLA achievement against new targets
Gather team feedback on target appropriateness
Adjust if necessary within first quarter



Example Capability-Specific Configuration:
sql-- Capability-level SLA targets
CAPABILITY_KEY               | RESPONSE_TIME_TARGET_DAYS | JUSTIFICATION
"DQ"                        | 3                         | Data quality issues require rapid response
"DE"                        | 5                         | Extract complexity requires standard timeframe  
"CC"                        | 7                         | Change approval process inherently longer
"RD"                        | 4                         | Reference data updates need quick turnaround
"RM"                        | 6                         | Records management standard process time
Business Day Calculations
Standard Business Hours

Monday - Friday: 9:00 AM to 5:00 PM (local time)
Weekends: Excluded by default
Holidays: Corporate holiday calendar integration
Time Zone: UTC standardization with local display

Business Day Logic Implementation
Power Query Implementation:
m// Standard business hours calculation (9 AM - 5 PM, weekdays only)
AddDurationBusiness = Table.AddColumn(AddDurationCalendar, "DurationBusinessHours", each
    let
        StartTime = [PreviousChangeTime],
        EndTime = [change_created],
        StartDate = Date.From(StartTime),
        EndDate = Date.From(EndTime),
        
        // Generate date range
        DateList = List.Dates(StartDate, Duration.Days(EndDate - StartDate) + 1, #duration(1,0,0,0)),
        
        // Filter for business days (exclude weekends and holidays)
        BusinessDays = List.Select(DateList, each 
            Date.DayOfWeek(_, Day.Monday) < 5 and  
            not List.Contains(HolidayCalendar[Date], _)
        ),
        
        // Calculate business hours for each day
        BusinessHours = List.Accumulate(BusinessDays, 0, (total, current) =>
            if current = StartDate and current = EndDate then
                // Same day calculation (9 AM - 5 PM)
                let 
                    ActualStart = Number.Max(Time.Hour(DateTime.Time(StartTime)), 9),
                    ActualEnd = Number.Min(Time.Hour(DateTime.Time(EndTime)), 17)
                in total + Number.Max(ActualEnd - ActualStart, 0)
            else if current = StartDate then
                // First day - from start time to 5 PM
                let ActualStart = Number.Max(Time.Hour(DateTime.Time(StartTime)), 9)
                in total + Number.Max(17 - ActualStart, 0)
            else if current = EndDate then
                // Last day - from 9 AM to end time  
                let ActualEnd = Number.Min(Time.Hour(DateTime.Time(EndTime)), 17)
                in total + Number.Max(ActualEnd - 9, 0)
            else
                // Full business day (8 hours)
                total + 8
        )
    in
        Number.Max(BusinessHours, 0)  // Ensure non-negative
)
SLA Performance Measurement
Core Calculation Logic
ResolutionTimeDays Implementation:
daxResolutionTimeDays = 
VAR CreatedDate = Fact_Ticket_Summary[created]
VAR ResolvedDate = Fact_Ticket_Summary[resolution_date]
VAR CalculationResult = 
    DATEDIFF(
        CreatedDate,
        COALESCE(ResolvedDate, NOW()),
        DAY
    )
RETURN 
    MAX(CalculationResult, 0)  // Ensure non-negative values
Met_SLA Logic:
daxMet_SLA = 
VAR ActualResolutionDays = Fact_Ticket_Summary[ResolutionTimeDays]
VAR IsResolved = Fact_Ticket_Summary[resolution_date] <> BLANK()
VAR SLA_Target = [SLA_Target_Days]

RETURN 
    SWITCH(
        TRUE(),
        NOT IsResolved, BLANK(),                    // Cannot determine yet
        ActualResolutionDays <= SLA_Target, TRUE,  // Met SLA
        ActualResolutionDays > SLA_Target, FALSE   // Missed SLA
    )
SLA Achievement Rate:
daxSLO_Achievement_Rate = 
VAR ResolvedTickets = 
    FILTER(
        Fact_Ticket_Summary,
        [resolution_date] <> BLANK() &&
        [Met_SLA] <> BLANK()
    )
VAR TotalResolved = COUNTROWS(ResolvedTickets)
VAR WithinSLO = 
    COUNTROWS(
        FILTER(ResolvedTickets, [Met_SLA] = TRUE)
    )
RETURN DIVIDE(WithinSLO, TotalResolved, 0) * 100
Basic Validation and Testing
Essential Test Scenarios
SLA Calculation Validation:
dax// Test SLA hierarchy resolution
Test_SLA_Hierarchy = 
VAR TestScenarios = 
    DATATABLE(
        "TicketKey", STRING,
        "IssueType", STRING,
        "CapabilityKey", STRING,
        "ExpectedSource", STRING,
        {
            ("TEST-001", "Bug", "DQ", "Capability"),
            ("TEST-002", "NewType", "DE", "Default"),
            ("TEST-003", "Task", null, "Default")
        }
    )
VAR ValidationResults = 
    ADDCOLUMNS(
        TestScenarios,
        "ActualSLA", [SLA_Target_Days],
        "SLASource", 
            IF(
                RELATED(Dim_Capability[ResponseTimeTargetDays]) <> BLANK(),
                "Capability",
                "Default"
            ),
        "TestPassed", [ExpectedSource] = [SLASource]
    )
RETURN ValidationResults
Business Day Validation:
dax// Validate business day calculations
Test_Business_Days = 
VAR TestCases = 
    DATATABLE(
        "StartDate", DATETIME,
        "EndDate", DATETIME,
        "ExpectedBusinessDays", INTEGER,
        {
            (#datetime(2024,1,1,9,0,0), #datetime(2024,1,3,17,0,0), 16),
            (#datetime(2024,1,5,14,0,0), #datetime(2024,1,8,11,0,0), 5),
            (#datetime(2024,1,6,10,0,0), #datetime(2024,1,7,15,0,0), 0)
        }
    )
VAR ValidationResults = 
    ADDCOLUMNS(
        TestCases,
        "ActualBusinessDays", [CalculatedBusinessHours],
        "TestPassed", ABS([ExpectedBusinessDays] - [CalculatedBusinessHours]) <= 1
    )
RETURN ValidationResults
Data Quality Checks
SLA Target Validation:

All capability SLA targets are positive numbers
Default SLA table covers all common issue types
No orphaned tickets without SLA targets
Business day calculations handle edge cases (holidays, weekends)

Configuration Management
Change Categories
Minor SLA Changes (no approval required):

Capability-level target adjustments within ±20% of current value
Default SLA table updates for new issue types
Business day rule clarifications

Major SLA Changes (approval required):

Capability-level target changes >20%
New capability onboarding with custom targets
Fundamental changes to business day calculations

Change Process
Standard Changes:

Request Submission via capability owner through Confluence
Impact Assessment by technical team (1-2 business days)
Implementation during next nightly sync
Validation and stakeholder notification

Change Tracking:
yamlCHANGE_AUDIT:
  change_type: "SLA_TARGET_UPDATE"
  capability_key: "DQ"
  old_value: 3
  new_value: 2
  justification: "Improved process automation reduces resolution time"
  requested_by: "capability.owner@company.com"
  approved_by: "system.admin@company.com"
  implemented_date: "2024-01-15T02:00:00Z"
Implementation Guidelines
Configuration Best Practices
Target Setting Guidelines:

Start conservative: Set targets you can achieve 90-95% of the time
Use historical data: Base targets on 80th percentile of historical performance
Consider capacity: Ensure targets align with available resources
Plan for review: Schedule quarterly target assessments

Common Configuration Patterns:
markdownRECOMMENDED STARTING POINTS:
- Data Quality capabilities: 2-3 days (rapid response needed)
- Data Engineering: 3-5 days (technical complexity)
- Change Controls: 5-7 days (approval processes)
- Administrative: 3-5 days (standard processing)
Avoiding Common Pitfalls
SLA Anti-Patterns:

Setting targets too aggressively (leads to gaming behavior)
Ignoring capacity constraints (creates unsustainable pressure)
Not aligning targets with customer expectations (creates service gaps)
Treating all issue types identically (ignores complexity differences)

Success Indicators:

Consistent SLA achievement rates (90-95%)
Improving team satisfaction with target appropriateness
Reduced escalations and customer complaints
Stable performance over time

Continuous Improvement
Monthly Reviews:

Track SLA achievement trends
Identify capability-specific performance patterns
Gather stakeholder feedback on target appropriateness
Adjust targets based on process improvements

Quarterly Assessments:

Comprehensive review of all SLA targets
Cross-capability benchmarking
Business alignment validation
Capacity planning integration

Getting Support
Configuration Questions:

Email: platform.support@company.com
Documentation: Capability-specific Confluence configuration pages
Training: Monthly SLA configuration workshops

Technical Issues:

Troubleshooting: System validation tools in Power BI
Urgent Issues: Contact platform support during business hours
After-hours: Check system status page for known issues


Key Takeaways
SLA customization in the simplified system focuses on:

Two-tier hierarchy: Capability → Default fallback
Business-driven targets: Based on operational capacity and customer needs
Continuous improvement: Regular review and adjustment cycles
Simplicity: Easy to understand, configure, and maintain

This streamlined approach ensures reliable SLA management that directly supports the six core KPIs while avoiding unnecessary complexity that can obscure performance insights.