# Frequently Asked Questions

## Getting Started

### Q: How do I get access to the SLO dashboard?
**A:** Access is provided through multiple channels:
- **SharePoint embedding** on your team's site or corporate intranet
- **Email reports** sent monthly on the 1st business day
- **Power BI mobile app** with full dashboard functionality
- **Microsoft Teams integration** for team channels and alerts

To request access, email **platform.support@company.com** with your team name, primary contact, and brief description of services.

### Q: What if I don't see any data when I first access the dashboard?
**A:** This is normal for new implementations:
- **New capabilities** need 2-4 weeks to accumulate meaningful trend data
- **Historical data** is loaded automatically going back 6 months from activation
- **Ticket categorization** must be properly configured in Jira
- **Permissions** may limit visibility to certain capabilities

Contact support if dashboards remain empty after one week of expected activity.

### Q: How often is data updated?
**A:** The system refreshes on a predictable schedule:
- **Nightly refresh** at 2:00 AM UTC with previous day's data
- **Historical trends** maintain a rolling 6-month window
- **Manual refresh** available for urgent updates (requires IT approval)
- **Real-time alerts** trigger immediately when thresholds are breached

All dashboards show the last refresh timestamp in the footer.

### Q: How long does onboarding take and what's involved?
**A:** Complete onboarding typically takes 4-6 weeks:
- **Week 1**: Initial request, kickoff meeting, capability definition
- **Weeks 2-3**: Service mapping, workflow analysis, SLO target setting
- **Week 4**: Configuration validation, team training
- **Weeks 5-6**: Fine-tuning, go-live, initial optimization

Most effort involves documenting your team's services and setting realistic performance targets.

---

## Understanding SLOs and KPIs

### Q: What's the difference between lead time, cycle time, and response time?
**A:** These measure different aspects of service delivery:
- **Lead Time**: Speed of initial response (ticket creation to work begins)
- **Cycle Time**: Efficiency of active work (work begins to completion)
- **Response Time**: Complete customer experience (creation to final resolution)

**Example**: A ticket created Monday, started Wednesday, completed Friday has:
- Lead Time: 2 days, Cycle Time: 2 days, Response Time: 4 days

### Q: What are the other KPIs beyond time-based SLOs?
**A:** The dashboard tracks multiple performance dimensions:
- **Throughput**: Volume of completed tickets (capacity planning)
- **Service Quality**: Accuracy and effectiveness of resolutions
- **Issue Resolution**: Resolution stability (tickets that stay resolved)
- **Reopened Rate**: Percentage of tickets requiring additional work

These provide a balanced view beyond just speed metrics.

### Q: How are weekends and holidays handled?
**A:** By default, time calculations use business days only:
- **Excludes weekends** (Saturday and Sunday)
- **Business hours**: 9 AM to 5 PM for partial days
- **Holiday calendar**: Configurable for your organization
- **Pause on waiting**: Timer stops for "Waiting for Customer" status

Teams can customize these rules based on their operational model.

### Q: Can SLO targets be changed after setup?
**A:** Yes, targets are adjustable through proper change management:
- **Capability owners** update targets via Confluence configuration pages
- **New tickets** use updated targets; existing tickets keep original SLAs
- **Approval required** for changes >25% to ensure business alignment
- **Quarterly reviews** recommended to align targets with operational reality

---

## Using Dashboards

### Q: Can I customize my dashboard view?
**A:** Several customization options are available:
- **Personal filters** and preferences saved per user
- **Role-based views** (Executive, Capability Owner, Team Member)
- **Mobile-optimized layouts** with key metrics highlighted
- **Subscription preferences** for reports and alerts

For major customization needs, submit a request for custom dashboard development.

### Q: Why do my numbers differ from Jira reports?
**A:** Differences typically stem from:
- **Business day calculations**: SLO dashboard excludes weekends/holidays
- **Active ticket filtering**: Only tickets with active=1 are included
- **Status timing rules**: Specific definitions for when timers start/stop
- **Aggregation periods**: Different grouping methods than standard Jira

For significant discrepancies, request a data validation review.

### Q: How do I export data for analysis?
**A:** Multiple export options support different use cases:
- **PDF reports**: Automatically generated monthly summaries
- **Excel export**: Available from individual dashboard visuals
- **CSV download**: For detailed ticket-level analysis
- **Power BI datasets**: For advanced analysis in Excel or other tools

API access is available for programmatic data extraction.

### Q: What does "tickets at risk" mean?
**A:** These are open tickets approaching SLO breach:
- **Calculation**: Tickets at 80%+ of their SLO target
- **Purpose**: Early warning system for proactive intervention
- **Action needed**: Review workload, reassign resources, or communicate delays
- **Alert triggers**: Configurable thresholds for automatic notifications

---

## Configuration and Setup

### Q: How do I set up or change SLO targets?
**A:** Configuration is managed through Confluence:
1. Navigate to your capability's configuration page
2. Update SLO targets table with new values
3. Include business justification for changes
4. Changes sync automatically during next nightly refresh
5. Monitor performance against new targets

Significant changes require approval from your capability owner.

### Q: How do I add new issue types or services?
**A:** Expansion follows a structured process:
- **Issue types**: Add to capability mapping table in Confluence
- **New services**: Define service names, automation levels, typical effort
- **Default SLAs**: System applies organizational defaults until custom targets set
- **Validation**: Automatic checks ensure mappings are complete and valid

### Q: How does the Default SLA Table work?
**A:** This provides fallback values ensuring every ticket has an SLA:
- **Hierarchy**: Service-specific → Capability-level → Default table → 5-day fallback
- **Coverage**: Bug (3 days), Task (5 days), Epic (10 days), etc.
- **Purpose**: Prevents orphaned tickets without SLA targets
- **Customization**: Teams can override defaults with capability-specific targets

### Q: What happens during configuration changes?
**A:** The change process includes safeguards:
- **Validation**: Automatic checking of new configuration rules
- **Impact assessment**: Analysis of affected tickets and metrics
- **Audit trail**: Complete tracking of who changed what when
- **Rollback capability**: Ability to revert changes if issues arise
- **Notification**: Stakeholders informed of successful updates

---

## Alerts and Notifications

### Q: Why aren't I receiving expected alerts?
**A:** Troubleshoot notification delivery:
- **Subscription verification**: Check alert preferences in Confluence
- **Email filtering**: Look in spam/junk folders for system emails
- **Threshold settings**: Alerts may not trigger if thresholds are too permissive
- **Delivery method**: Confirm correct email address and notification channels

Use the "Test Alert" feature to verify your subscription is working.

### Q: How do I prevent alert fatigue?
**A:** The system includes several anti-fatigue features:
- **Intelligent suppression**: Prevents duplicate alerts for the same issue
- **Smart batching**: Groups related alerts into digestible summaries
- **Dynamic thresholds**: Self-adjusting based on your response patterns
- **Timing optimization**: Delivers alerts when you're most likely to act

Customize sensitivity and delivery timing to match your workflow.

### Q: Can alerts integrate with team communication tools?
**A:** Multiple integration options are supported:
- **Microsoft Teams**: Configure alerts for team channels
- **Email distribution lists**: Include team aliases in subscriptions
- **SMS notifications**: Available for critical alerts (admin setup required)
- **Slack integration**: Available in organizations using Slack

Set different thresholds for individual vs. team notifications.

---

## Technical and Performance

### Q: Which browsers and devices are supported?
**A:** The dashboard supports modern browsers and devices:
- **Recommended**: Microsoft Edge, Google Chrome (latest versions)
- **Supported**: Firefox, Safari (latest versions)
- **Mobile**: Power BI app on iOS/Android, responsive web interface
- **Requirements**: JavaScript enabled, cookies allowed

Internet Explorer is not supported. Ensure your browser is up to date.

### Q: What should I do if dashboards load slowly?
**A:** Performance optimization steps:
1. **Clear browser cache** and cookies for the site
2. **Reduce date ranges** for faster query processing
3. **Limit capability selection** to only relevant areas
4. **Check network connection** speed and stability
5. **Contact support** if problems persist after optimization

Peak usage times (Monday mornings) may affect response times.

### Q: How is my data protected and backed up?
**A:** Comprehensive data protection measures:
- **Automated backups**: Daily snapshots with 30-day retention
- **Geographic redundancy**: Data replicated across multiple centers
- **Access controls**: Role-based permissions and row-level security
- **Audit logging**: Complete tracking of data access and changes
- **Compliance**: Meets organizational data governance requirements

Recovery Time Objective (RTO) is less than 4 hours for critical functions.

---

## Mobile and Remote Access

### Q: How do I access dashboards on mobile devices?
**A:** Mobile access through multiple options:
- **Power BI mobile app**: Download from App Store/Google Play for best experience
- **Mobile web browser**: Responsive design works on all modern mobile browsers
- **Email reports**: Monthly PDFs optimized for mobile viewing
- **Teams mobile**: Quick alerts and status checks

Sign in with your organizational credentials for seamless access.

### Q: What features are available on mobile?
**A:** Full mobile functionality includes:
- **All dashboard views** with touch-optimized navigation
- **Interactive filtering** and drill-down capabilities
- **Alert management** and subscription preferences
- **Team performance monitoring** with real-time updates
- **Export capabilities** for sharing and offline viewing

Advanced configuration features require desktop access.

---

## Troubleshooting

### Q: Dashboard shows no data or missing information
**A:** Systematic troubleshooting approach:
1. **Verify permissions**: Confirm access to relevant capabilities
2. **Check date range**: Ensure selected timeframe contains activity
3. **Review filters**: Clear any applied filters that might hide data
4. **Validate configuration**: Ensure issue types properly mapped
5. **Browser refresh**: Hard refresh (Ctrl+F5) to reload data

Contact support with specific error messages and screenshots.

### Q: My calculations don't match expectations
**A:** Common calculation discrepancies:
- **Business day logic**: System excludes weekends while manual calculations may not
- **SLO hierarchy**: Check which SLA source is being used (service, capability, default)
- **Ticket status filtering**: Only active tickets (active=1) included in calculations
- **Time zone differences**: All calculations use UTC timestamps

Request data validation review for persistent inconsistencies.

### Q: I can't access certain capabilities or see limited data
**A:** Access limitations usually involve:
- **Row-level security**: Your permissions may be restricted to specific capabilities
- **Recent configuration changes**: Updates may have affected your access level
- **User role assignment**: Verify your role (Executive, Capability Owner, Team Member)
- **Capability mapping**: Ensure your user account is properly associated

Contact your capability owner or system administrator to review permissions.

---

## Administration and Management

### Q: How do I add new team members?
**A:** User provisioning process:
1. **Submit request** via service portal with business justification
2. **Specify access level** and capability scope required
3. **Await approval** from capability owner and system administrator
4. **Complete mandatory training** before full access activation
5. **Validate permissions** during first week of access

Capability owners can expedite requests for their direct team members.

### Q: What's the change management process?
**A:** Structured change approval ensures stability:
- **Standard changes**: Pre-approved, executed by capability owners
- **Major changes**: Require impact assessment and stakeholder approval
- **Emergency changes**: Expedited process with mandatory post-review
- **All changes**: Documented in audit system with rollback capability

See the change procedures guide for detailed workflows and approval matrices.

### Q: How long is data retained and can I access historical reports?
**A:** Data retention follows organizational policies:
- **Active dashboard data**: Rolling 6-month window for performance
- **Historical archives**: 7 years for compliance and trend analysis
- **Configuration changes**: Complete audit trail maintained indefinitely
- **Custom reports**: Available on request for specific time periods

Contact support for access to historical data beyond the standard 6-month window.

---

## Advanced Features

### Q: Can I create custom KPIs specific to my team's needs?
**A:** Custom KPI development is supported:
- **Business case required**: Must demonstrate specific value and need
- **Technical feasibility**: Assessment of data availability and complexity
- **Development timeline**: 2-4 weeks depending on complexity
- **Ongoing maintenance**: Custom KPIs require regular validation and updates

Submit custom KPI requests through the service portal with detailed requirements.

### Q: How do reopened tickets affect my metrics?
**A:** Reopened tickets are tracked separately for quality insights:
- **Detection**: Automatic identification of tickets returning to "Open" from "Done"
- **Quality impact**: Affects Service Quality KPI and first-pass resolution rates
- **Root cause analysis**: Helps identify process improvements needed
- **Trend tracking**: Monitor reopening patterns over time

Reopened rate is a key indicator of resolution effectiveness.

### Q: Is there API access for automated reporting?
**A:** Programmatic access supports automation:
- **Power BI REST API**: Standard endpoints for dashboard and dataset access
- **Custom APIs**: SLO-specific endpoints for specialized requirements
- **Authentication**: OAuth 2.0 and service account support
- **Rate limiting**: Appropriate throttling to maintain system performance

API documentation and credentials available through IT services portal.

---

## Getting Additional Help

If your question isn't answered here:

- **Email support**: platform.support@company.com
- **Service portal**: Submit detailed requests with screenshots
- **Teams channel**: #slo-dashboard-support for quick questions
- **Office hours**: Tuesdays 2-3 PM EST for live assistance
- **Emergency escalation**: Extension 5555 for urgent system-wide issues

For training sessions and capability onboarding, contact the Center of Excellence at coe@company.com.