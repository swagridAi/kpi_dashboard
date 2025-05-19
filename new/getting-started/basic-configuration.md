# Basic Configuration Guide

This guide covers the essential configuration steps to get your team's SLO tracking system working effectively. All configuration is done through your team's designated Confluence page, which automatically synchronizes with the dashboard system nightly.

**Important**: Configuration changes take effect after the next nightly data refresh. Plan accordingly for any time-sensitive adjustments.

## Essential Configuration Areas

### 1. SLO Targets

Setting realistic SLO targets is crucial for meaningful performance measurement. The system uses a hierarchical approach to determine which target applies to each ticket:

1. **Service-specific targets** (if you've set custom targets for specific services)
2. **Capability-level targets** (your team's general targets)
3. **Default organizational targets** (fallback values)

**Default Starting Points by Issue Type:**
- **Bug**: 3 business days (critical defects need faster response)
- **Task**: 5 business days (standard work items)
- **Story**: 7 business days (feature development work)
- **Epic**: 10 business days (large initiatives, complex work)
- **Incident**: 1 business day (urgent production issues)

**How to Configure in Confluence:**
1. Navigate to your team's SLO Configuration page
2. Update the "SLO Targets" table with your desired values
3. Provide business justification for each target
4. Save the page - changes sync automatically overnight

**Setting Realistic Targets:**
- Review historical performance over the past 3-6 months
- Set targets you can meet 90-95% of the time
- Consider seasonal variations and capacity constraints
- Start conservative - targets can be made more aggressive later

**Priority Adjustments:**
The system automatically adjusts targets based on priority:
- **P1 (Critical)**: Reduces target by 50%
- **P2 (High)**: Reduces target by 25%  
- **P3 (Medium)**: Uses standard target
- **P4 (Low)**: Extends target by 50%

### 2. Issue Type Mapping

Connect your Jira issue types to your capability and services. This mapping determines which SLO targets apply to each ticket.

**Configuration Steps in Confluence:**
1. **List Your Issue Types**: Document all Jira issue types your team uses
2. **Assign to Capability**: Map each issue type to your capability (e.g., Data Quality, Data Extracts)
3. **Define Services**: Group related issue types into logical services within your capability
4. **Specify Automation Level**: Mark each as Manual, Semi-Automated, or Fully Automated

**Example Issue Type Mapping:**
```
Issue Type: Bug
├── Capability: Data Quality
├── Service: Data Validation Service  
├── Automation Level: Manual
└── Typical Effort Hours: 8

Issue Type: Data Extract Request
├── Capability: Data Extracts
├── Service: Custom Extract Service
├── Automation Level: Semi-Automated
└── Typical Effort Hours: 4
```

**Best Practices:**
- Map issue types to the most specific service possible
- Be consistent with automation level classifications
- Update mappings when you introduce new issue types
- Include typical effort hours to help with capacity planning

### 3. Status Rules

Define how ticket statuses relate to SLO time measurements. This determines when the system starts and stops timing for lead time, cycle time, and response time calculations.

**Time Measurement Definitions:**
- **Lead Time**: How quickly you begin working (creation to "In Progress")
- **Cycle Time**: How efficiently you complete work ("In Progress" to "Done")  
- **Response Time**: Complete customer experience (creation to "Done")

**Standard Status Classifications:**
```
Status Type          │ When Timer...        │ Includes in...
────────────────────┼─────────────────────┼──────────────────
Backlog             │ Starts Lead Time     │ Lead Time
In Progress         │ Starts Cycle Time    │ Cycle Time
Code Review         │ Continues           │ Cycle Time
Waiting for Customer│ Pauses Timer        │ Neither (external)
Testing            │ Continues           │ Cycle Time  
Done/Resolved      │ Ends All Timers     │ Response Time End
```

**Business Rules Configuration:**
- **Exclude Weekends**: Calculations use business days only (Monday-Friday)
- **Pause on Waiting**: Timer stops for external dependencies
- **Holiday Calendar**: Organizational holidays are excluded from business day calculations

**Configuration in Confluence:**
Update your "Status Rules" table with the time type (lead/cycle/response/pause) for each status your team uses. The system will apply these rules automatically.

### 4. Alert Thresholds

Set up proactive alerts to manage SLO performance before issues become critical. Start with essential alerts and add more sophisticated notifications as needed.

**Essential Alert Types:**
- **SLO At Risk (80% threshold)**: Early warning when 80% of target time has elapsed
- **SLO Breach**: Immediate notification when target is exceeded
- **Daily Digest**: Summary of previous day's performance
- **Weekly Summary**: Comprehensive capability performance report

**Alert Configuration in Confluence:**
1. Navigate to your "Alert Preferences" section
2. Set thresholds for each alert type (recommended: start with 80% for at-risk)
3. Choose delivery methods (email, Teams, mobile push)
4. Select recipients (yourself, team leads, stakeholders)
5. Test alert delivery with sample notifications

**Delivery Method Guidelines:**
- **Email**: Daily summaries, weekly reports, non-urgent notifications
- **Microsoft Teams**: Real-time breaches, team-wide announcements
- **Mobile Push**: Critical alerts only (SLO breaches, major issues)
- **SMS**: Emergency escalation (optional, for critical capabilities)

**Anti-Fatigue Features:**
The system includes smart features to prevent alert overload:
- **Intelligent suppression**: Prevents duplicate alerts for the same issue
- **Smart batching**: Groups related alerts into single digests
- **Dynamic timing**: Sends alerts when you're most likely to act

## Configuration Best Practices

**Start with Defaults, Then Customize**
- Use organizational defaults as starting points
- Make incremental adjustments based on real data
- Change one configuration element at a time to understand impact

**Ground Decisions in Data**
- Review 3-6 months of historical ticket data before setting targets
- Adjust targets to achieve 90-95% success rate (not 100%)
- Use actual workflow patterns to define status rules

**Involve Your Team**
- Get input from team members on realistic targets
- Ensure status mappings reflect how work actually flows  
- Document configuration decisions and rationales

**Plan for Iteration**
- Schedule monthly configuration reviews for first quarter
- Move to quarterly reviews once configuration stabilizes
- Track changes with business justifications for audit purposes

**Maintain Documentation**
- Keep team-specific workflow documentation current
- Record all configuration changes with dates and reasons
- Share configuration decisions with new team members

## Validation Checklist

Before activating your configuration for live dashboards:

**Configuration Completeness:**
- [ ] SLO targets are set for all issue types your team uses
- [ ] All Jira issue types are mapped to appropriate capabilities and services  
- [ ] Status rules are defined for your complete workflow
- [ ] Alert thresholds are configured with appropriate recipients
- [ ] Business justifications are documented for all targets

**Technical Validation:**
- [ ] Test tickets calculate correctly using new status rules
- [ ] Sample SLO calculations match expected results
- [ ] Alert test messages reach intended recipients successfully
- [ ] Dashboard displays expected data after configuration sync

**Team Preparation:**
- [ ] Team members understand the new SLO definitions
- [ ] Workflow changes are communicated to all stakeholders
- [ ] First month review meeting is scheduled
- [ ] Documentation is accessible to all team members
- [ ] Escalation procedures are established for configuration issues

## Quick Setup Checklist

For teams ready to start immediately:

**Week 1:**
- [ ] Complete capability definition in Confluence
- [ ] Map all issue types to services
- [ ] Set initial SLO targets using defaults
- [ ] Configure basic status rules

**Week 2:**
- [ ] Set up breach alerts for capability owner
- [ ] Test configuration with sample tickets
- [ ] Train team on SLO definitions
- [ ] Validate initial dashboard data

**Week 3:**
- [ ] Refine targets based on initial data
- [ ] Add team member alert subscriptions
- [ ] Document team-specific workflows
- [ ] Schedule first performance review

## Next Steps

Once basic configuration is complete:

- Review [Team Operations Guide](../dashboard-usage/team-operations.md) for daily usage
- Explore [Custom KPIs](../advanced-configuration/custom-kpis.md) for specialized measurements
- Set up [Alerts and Subscriptions](../dashboard-usage/alerts-and-subscriptions.md) for your team

## Getting Help

**Configuration Questions:**
- Email: [platform.support@company.com](mailto:platform.support@company.com)
- Documentation: Your team's Confluence SLO Configuration page includes detailed setup guides
- Training: Monthly capability owner sessions include configuration workshops

**Technical Issues:**
- Troubleshooting: See [Troubleshooting Guide](../administration/troubleshooting.md) for common problems
- Urgent Issues: Contact platform support during business hours (Ext. 5555)  
- After-hours: Check system status page for known issues

**Peer Support:**
- Monthly Capability Owner meetings (first Tuesday of each month)
- Internal forums: #slo-dashboard-help on Teams
- Champions network: Experienced users available for mentoring

Remember: Configuration is iterative. Start with basics, learn from your data, and refine over time to create an SLO system that truly supports your team's success.