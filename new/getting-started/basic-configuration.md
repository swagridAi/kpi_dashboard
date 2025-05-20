Basic Configuration Guide
This guide covers the essential configuration steps to get your team's SLO tracking system working effectively. All configuration is done through your team's designated Confluence page, which automatically synchronizes with the dashboard system nightly.
Important: Configuration changes take effect after the next nightly data refresh. Plan accordingly for any time-sensitive adjustments.
Essential Configuration Areas
1. SLO Targets
Setting realistic SLO targets is crucial for meaningful performance measurement. The system uses a simple 2-tier approach to determine which target applies to each ticket:

Capability-level targets (your team's configured targets)
Default organizational targets (automatic fallback values)

Default Starting Points by Issue Type:

Bug: 3 business days (critical defects need faster response)
Task: 5 business days (standard work items)
Story: 7 business days (feature development work)
Epic: 10 business days (large initiatives, complex work)
Incident: 1 business day (urgent production issues)

How to Configure in Confluence:

Navigate to your team's SLO Configuration page
Update the "SLO Targets" table with your desired values
Provide business justification for each target
Save the page - changes sync automatically overnight

Setting Realistic Targets:

Review historical performance over the past 3-6 months
Set targets you can meet 90-95% of the time
Consider seasonal variations and capacity constraints
Start conservative - targets can be made more aggressive later

2. Issue Type Mapping
Connect your Jira issue types to your capability. This mapping determines which SLO targets apply to each ticket and ensures proper categorization for the 6 core KPIs.
Configuration Steps in Confluence:

List Your Issue Types: Document all Jira issue types your team uses
Assign to Capability: Map each issue type to your capability (e.g., Data Quality, Data Extracts)
Specify Automation Level: Mark each as Manual, Semi-Automated, or Fully Automated (for Throughput analysis)

Example Issue Type Mapping:
Issue Type: Bug
├── Capability: Data Quality
├── Automation Level: Manual
└── Notes: Data quality defects and corrections

Issue Type: Data Extract Request
├── Capability: Data Extracts
├── Automation Level: Semi-Automated
└── Notes: Custom and standard data extractions
Best Practices:

Map issue types to the most appropriate capability
Be consistent with automation level classifications
Update mappings when you introduce new issue types
Ensure all team issue types are included to avoid orphaned tickets

3. Status Rules
Define how ticket statuses relate to the three core time measurements: Lead Time, Cycle Time, and Response Time. This determines when the system starts and stops timing for each KPI.
Time Measurement Definitions:

Lead Time: How quickly you begin working (creation to "In Progress")
Cycle Time: How efficiently you complete work ("In Progress" to "Done")
Response Time: Complete customer experience (creation to "Done")

Standard Status Classifications:
Status Type          │ Time Measurement     │ Business Purpose
────────────────────┼─────────────────────┼──────────────────────────
Backlog             │ Lead Time Start      │ Measures response speed
New/Open            │ Lead Time Start      │ Measures response speed
In Progress         │ Cycle Time Start     │ Measures work efficiency
Development         │ Cycle Time          │ Measures work efficiency
Testing/Review      │ Cycle Time          │ Measures work efficiency
Done/Resolved       │ Response Time End    │ Measures total delivery
Business Day Configuration:

Exclude Weekends: Calculations use business days only (Monday-Friday)
Business Hours: Standard 9 AM to 5 PM for partial-day calculations
Holiday Calendar: Organizational holidays are excluded from business day calculations

Configuration in Confluence:
Update your "Status Rules" table with the time measurement type (lead/cycle/response/other) for each status your team uses. The system will apply these rules automatically to calculate your core KPIs.
4. Alert Thresholds
Set up proactive alerts to manage SLO performance before issues become critical. Focus on essential notifications that support the 6 core KPIs.
Essential Alert Types:

SLO At Risk (80% threshold): Early warning when 80% of target time has elapsed
SLO Breach: Immediate notification when target is exceeded
Daily Digest: Summary of previous day's performance across all 6 KPIs
Weekly Summary: Comprehensive capability performance report

Alert Configuration in Confluence:

Navigate to your "Alert Preferences" section
Set thresholds for each alert type (recommended: start with 80% for at-risk)
Choose delivery methods (email, Teams, mobile push)
Select recipients (yourself, team leads, stakeholders)
Test alert delivery with sample notifications

Delivery Method Guidelines:

Email: Daily summaries, weekly reports, non-urgent notifications
Microsoft Teams: Real-time breaches, team-wide announcements
Mobile Push: Critical alerts only (SLO breaches, major issues)

Anti-Fatigue Features:
The system includes smart features to prevent alert overload:

Intelligent suppression: Prevents duplicate alerts for the same issue
Smart batching: Groups related alerts into single digests
Dynamic timing: Sends alerts when you're most likely to act

Configuration Best Practices
Start with Defaults, Then Customize

Use organizational defaults as starting points
Make incremental adjustments based on real data
Change one configuration element at a time to understand impact

Ground Decisions in Data

Review 3-6 months of historical ticket data before setting targets
Adjust targets to achieve 90-95% success rate (not 100%)
Use actual workflow patterns to define status rules

Involve Your Team

Get input from team members on realistic targets
Ensure status mappings reflect how work actually flows
Document configuration decisions and rationales

Plan for Iteration

Schedule monthly configuration reviews for first quarter
Move to quarterly reviews once configuration stabilizes
Track changes with business justifications for audit purposes

Maintain Documentation

Keep team-specific workflow documentation current
Record all configuration changes with dates and reasons
Share configuration decisions with new team members

Validation Checklist
Before activating your configuration for live dashboards:
Configuration Completeness:

 SLO targets are set for all issue types your team uses
 All Jira issue types are mapped to your capability
 Status rules are defined for lead time, cycle time, and response time measurements
 Alert thresholds are configured with appropriate recipients
 Business justifications are documented for all targets

Core KPI Support:

 Lead Time measurement properly configured (creation to work start)
 Cycle Time measurement covers active work period
 Response Time captures complete customer experience
 Throughput calculation includes all completed work
 Service Quality tracking covers SLO achievement
 Issue Resolution Time uses standard resolution definitions

Technical Validation:

 Test tickets calculate correctly using new status rules
 Sample SLO calculations match expected results
 Alert test messages reach intended recipients successfully
 Dashboard displays expected data after configuration sync

Team Preparation:

 Team members understand the 6 core KPI definitions
 Workflow changes are communicated to all stakeholders
 First month review meeting is scheduled
 Documentation is accessible to all team members
 Escalation procedures are established for configuration issues

Quick Setup Checklist
For teams ready to start immediately:
Week 1:

 Complete capability definition in Confluence
 Map all issue types to your capability
 Set initial SLO targets using organizational defaults
 Configure basic status rules for the 3 time measurements

Week 2:

 Set up breach alerts for capability owner
 Test configuration with sample tickets
 Train team on the 6 core KPI definitions
 Validate initial dashboard data

Week 3:

 Refine targets based on initial data
 Add team member alert subscriptions
 Document team-specific workflows
 Schedule first performance review

Next Steps
Once basic configuration is complete:

Review Understanding Your First Dashboard for interpreting your 6 core KPIs
Explore Team Operations Guide for daily usage patterns
Set up Alerts and Subscriptions for your team

Getting Help
Configuration Questions:

Email: platform.support@company.com
Documentation: Your team's Confluence SLO Configuration page includes detailed setup guides
Training: Monthly capability owner sessions include configuration workshops

Technical Issues:

Troubleshooting: See Troubleshooting Guide for common problems
Urgent Issues: Contact platform support during business hours (Ext. 5555)
After-hours: Check system status page for known issues

Peer Support:

Monthly Capability Owner meetings (first Tuesday of each month)
Internal forums: #slo-dashboard-help on Teams
Champions network: Experienced users available for mentoring

Remember: Configuration is iterative. Start with the essentials for the 6 core KPIs, learn from your data, and refine over time to create an SLO system that truly supports your team's success.