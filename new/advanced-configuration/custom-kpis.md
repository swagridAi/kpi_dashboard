Custom KPI Configuration
Understanding the 6 Core KPIs
The SLO Dashboard employs a focused set of six key performance indicators that provide comprehensive service delivery insights while maintaining simplicity and clarity. These KPIs are designed to measure what matters most without overwhelming complexity.

Why Six KPIs?
Each KPI reveals different aspects of service performance:

Time-based metrics (Lead, Cycle, Response Time) show how work flows through your process
Volume metrics (Throughput) reveal capacity and delivery consistency
Quality metrics (Service Quality, Issue Resolution Time) ensure speed doesn't sacrifice effectiveness
Understanding these KPIs helps teams measure what matters most to their customers while providing the data needed for continuous improvement.

KPI Architecture Overview
The system uses a simplified dimensional model optimized for the six core KPIs:

Fact Tables:

Fact_Ticket_Status_Change: Granular status transitions with duration calculations
Fact_Ticket_Summary: Aggregated ticket-level metrics with SLO achievement flags
Dimension Tables:

Dim_Date: Business calendar with holiday support
Dim_Capability: Service capabilities with SLO targets
Dim_Status: Ticket statuses with time-type classifications
Configuration:

Config_Issue_Type_Capability_Mapping: Maps issue types to capabilities
Default_SLA_Table: Fallback SLA definitions
All KPIs automatically inherit global filtering by capability, time period, and issue type, with consistent data refresh schedules and validation rules.

Core KPI Implementation
Time-Based KPIs
Lead Time
Business Definition: How quickly your team begins working on new requests

Implementation:

dax
Lead_Time_Days = 
VAR LeadTimeHours = 
    CALCULATE(
        AVERAGE(Fact_Ticket_Status_Change[DurationBusinessHours]),
        Fact_Ticket_Status_Change[IsLeadTimeStart] = TRUE
    )
RETURN DIVIDE(LeadTimeHours, 24, 0)
Configuration Notes:

Measured from ticket creation to first "In Progress" status
Uses business hours only (excludes weekends and holidays)
Typical targets: 1-2 days for standard work
Cycle Time
Business Definition: How efficiently you complete work once started

Implementation:

dax
Cycle_Time_Days = 
VAR CycleTimeTickets = 
    SUMMARIZE(
        FILTER(Fact_Ticket_Status_Change, [IsCycleTimeStart] = TRUE),
        [TicketKey],
        "CycleTime", SUM(Fact_Ticket_Status_Change[DurationBusinessHours])
    )
RETURN DIVIDE(AVERAGEX(CycleTimeTickets, [CycleTime]), 24, 0)
Configuration Notes:

Measured from work start to completion
Excludes waiting states (blocked, waiting for customer)
Typical targets: 2-5 days depending on work complexity
Response Time
Business Definition: Total customer experience from request to delivery

Implementation:

dax
Response_Time_Days = 
CALCULATE(
    AVERAGE(Fact_Ticket_Summary[ResolutionTimeDays]),
    Fact_Ticket_Summary[IsResolved] = TRUE
)
Configuration Notes:

Complete end-to-end measurement
Forms the basis for SLA commitments
Typical targets: 3-7 days for standard requests
Volume KPI
Throughput
Business Definition: Volume of tickets completed over time

Implementation:

dax
Throughput = 
CALCULATE(
    COUNTROWS(Fact_Ticket_Summary),
    Fact_Ticket_Summary[IsResolved] = TRUE,
    USERELATIONSHIP(Fact_Ticket_Summary[ResolvedDate], Dim_Date[Date])
)
Configuration Notes:

Counts completed tickets in the selected time period
Used for capacity planning and trend analysis
Automatically respects capability and date filters
Quality KPIs
Service Quality
Business Definition: Percentage of tickets resolved within SLA timeframes

Implementation:

dax
Service_Quality_KPI = 
VAR ResolvedTickets = 
    FILTER(
        Fact_Ticket_Summary,
        [IsResolved] = TRUE && NOT ISBLANK([Met_SLA])
    )
VAR TicketsMetSLA = 
    FILTER(ResolvedTickets, [Met_SLA] = TRUE)
RETURN
    DIVIDE(COUNTROWS(TicketsMetSLA), COUNTROWS(ResolvedTickets), 0) * 100
SLA Target Resolution (Simplified 2-tier hierarchy):

dax
SLA_Target_Days = 
COALESCE(
    -- Priority 1: Capability-level target
    RELATED(Dim_Capability[ResponseTimeTargetDays]),
    -- Priority 2: Default fallback
    5
)
Configuration Notes:

Uses simplified capability-level SLA targets only
No priority adjustments or service-specific overrides
Target: 90-95% achievement rate
Issue Resolution Time
Business Definition: Average time from creation to final resolution

Implementation:

dax
Issue_Resolution_Time = 
CALCULATE(
    AVERAGE(Fact_Ticket_Summary[ResolutionTimeDays]),
    Fact_Ticket_Summary[IsResolved] = TRUE
)
Configuration Notes:

Simple average resolution time in calendar days
Provides baseline understanding of delivery speed
Use alongside Service Quality for complete picture
Implementation Best Practices
Planning Your KPI Setup
Requirements Gathering:

Business Question: What specific business question does this KPI measurement answer?
Success Metrics: How will you measure if the KPI tracking is valuable?
Data Availability: Confirm all required source data exists in Jira
Stakeholder Alignment: Ensure capability owners understand and accept the metrics
Technical Implementation
Data Model Considerations:

Leverage existing fact and dimension tables
Pre-calculated columns optimize query performance
Business day calculations maintained in Power Query layer
Configuration Process:

Set SLO Targets: Update capability-level targets in Confluence
Map Issue Types: Ensure all ticket types properly categorized
Define Status Rules: Configure lead/cycle/response time boundaries
Validate Calculations: Test with known ticket examples
Performance Optimization
Query Performance:

Use calculated columns for frequently accessed computations
Leverage Power BI's VertiPaq compression
Implement appropriate relationship directions
Data Refresh:

Standard nightly refresh handles all six KPIs
Incremental refresh for large historical datasets
Validate data quality after each refresh
Validation and Testing
KPI Accuracy Validation
Core Validation Tests:

dax
// Test SLA calculation accuracy
SLA_Validation_Test = 
VAR TestScenarios = 
    DATATABLE(
        "TicketKey", STRING,
        "ActualDays", INTEGER,
        "TargetDays", INTEGER,
        "ExpectedResult", BOOLEAN,
        {
            ("TEST-001", 2, 3, TRUE),   // Met SLA
            ("TEST-002", 5, 3, FALSE),  // Missed SLA
            ("TEST-003", 3, 3, TRUE)    // Exactly met SLA
        }
    )
VAR ValidationResults = 
    ADDCOLUMNS(
        TestScenarios,
        "Actual_Met_SLA", 
            CALCULATE(
                MAX(Fact_Ticket_Summary[Met_SLA]),
                Fact_Ticket_Summary[key] = [TicketKey]
            ),
        "Test_Result", [ExpectedResult] = [Actual_Met_SLA]
    )
RETURN ValidationResults
Data Quality Checks:

Verify ticket counts match Jira exports
Validate business day calculations
Confirm SLA target application
Test time zone consistency
Business Validation
Key Validation Points:

Historical Accuracy: Sample manual calculations match dashboard results
Business Logic: Status transitions correctly identify time boundaries
Capability Mapping: All team tickets assigned to correct capability
SLA Reasonableness: Targets reflect actual business commitments
User Acceptance Testing
Test Scenarios:

Capability filtering shows only relevant tickets
Time period selections update all KPIs correctly
Export functionality maintains data integrity
Mobile access displays KPIs appropriately
Maintenance and Evolution
Monthly KPI Review
Performance Assessment:

Review KPI trends for unusual patterns
Validate SLA targets remain appropriate
Check for data quality issues
Assess user adoption and feedback
Continuous Improvement
Enhancement Process:

Gather Requirements: Document specific business needs
Evaluate Feasibility: Assess technical and resource requirements
Impact Analysis: Consider effects on existing functionality
Stakeholder Approval: Obtain consensus before implementation
Troubleshooting Common Issues
KPI Calculation Discrepancies:

Verify business day logic settings
Check capability mapping completeness
Validate time zone configurations
Review status classification rules
Performance Issues:

Monitor calculation timing during refresh
Optimize DAX expressions if needed
Consider aggregation strategies for large datasets
Next Steps
Ready to implement these KPIs?

Configure Your Capability: Set SLO targets in Confluence → Basic Configuration
Use Daily Operations: Integrate KPIs into workflow → Team Operations
Set Up Monitoring: Configure alerts and subscriptions → Alerts and Subscriptions
Understanding these six KPIs provides the foundation for data-driven service improvement. They offer the common language needed to discuss performance, identify opportunities, and demonstrate value across your organization.

For technical implementation assistance or questions about specific KPI configurations, contact the Technical Analytics team through the SLO Enhancement portal.

