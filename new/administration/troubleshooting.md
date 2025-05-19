# Troubleshooting Guide

## Quick Diagnostic Flowchart

```
Issue Reported
     ↓
[Can access main dashboard?] → NO → Check Access & Permissions
     ↓ YES
[Data showing/updating?] → NO → Check Data & Refresh Issues  
     ↓ YES
[Calculations correct?] → NO → Check Configuration Issues
     ↓ YES
[Alerts working?] → NO → Check Notification Issues
     ↓ YES
[Performance acceptable?] → NO → Check Performance Issues
     ↓ YES
Review Advanced/Integration Issues
```

## Issue Severity Matrix

| **Severity** | **Response Time** | **Examples** | **Escalation** |
|--------------|------------------|--------------|----------------|
| **Critical** | Immediate | System down, data corruption, security breach | L2 + Management |
| **High** | 4 hours | Major functionality broken, calculation errors | L2 Support |
| **Medium** | 24 hours | Individual user issues, minor functionality | L1 Support |
| **Low** | 72 hours | Enhancement requests, cosmetic issues | L1 Support |

---

## Common Issue Categories

### 1. Access & Permission Issues

#### Cannot Access Dashboard
**Diagnostic Questions:**
- Can you log into Power BI service at all?
- Do you see the dashboard in your workspace list?
- What error message appears?

**Quick Resolution Steps:**
1. Verify login credentials and MFA setup
2. Check browser (Chrome/Edge latest versions)
3. Clear cache: `Ctrl+Shift+Delete` → Clear all
4. Try incognito/private mode
5. Test different network connection

**Common Causes & Fixes:**
- **"Access Denied"** → Contact capability owner for permission
- **"Not Found"** → Dashboard may be renamed/moved
- **Login Loop** → Clear all cookies, restart browser
- **MFA Issues** → Re-register authentication app

#### Limited Data Visibility
**Diagnostic Questions:**
- Which capabilities can you see vs. expect to see?
- Are you filtered to specific time periods?
- Has your role recently changed?

**Resolution:**
1. Check active filters (clear all filters button)
2. Expand date range to last 6 months
3. Verify capability team membership
4. Request cross-capability access if needed

### 2. Data & Refresh Issues

#### Dashboard Shows "No Data" or Outdated Data
**Diagnostic Questions:**
- What's the "Last Refreshed" timestamp?
- Are you looking at the right time period?
- Does the issue affect all users or just you?

**Resolution Steps:**
1. **Check Refresh Status:**
   - Look for refresh timestamp on dashboard
   - Normal refresh: Daily at 2 AM local time
   - If >24 hours old, escalate immediately

2. **Validate Time Filters:**
   - Reset to "Last 6 months"
   - Check for hidden filters
   - Verify business day/calendar settings

3. **Source Data Verification:**
   - Confirm tickets exist in Jira for period
   - Check if new projects/issue types added
   - Validate service account permissions

#### Specific Tickets Missing
**Troubleshooting Checklist:**
```
□ Ticket exists in Jira with active=1
□ Created within data extraction window  
□ Project included in data source query
□ Issue type mapped to capability
□ Status changes recorded in changelog
```

**Resolution:**
1. Cross-reference ticket in Jira directly
2. Check issue type to capability mapping
3. Verify ticket creation date within range
4. Escalate if ticket meets all criteria but still missing

### 3. Calculation & Configuration Issues

#### SLO Percentages Seem Wrong
**Diagnostic Questions:**
- Are SLO targets recently changed?
- Do manual calculations match dashboard?
- Are business day rules applied correctly?

**Validation Process:**
1. **Sample Calculation Check:**
   - Pick 5 recent resolved tickets
   - Calculate SLO manually (resolution time vs. target)
   - Compare with dashboard values

2. **Configuration Verification:**
   - Check SLO targets in Confluence
   - Verify issue type mappings
   - Confirm business day exclusions

3. **Business Rules Check:**
   - Weekend exclusions enabled correctly
   - Holiday calendar configured
   - Time zone settings consistent

#### Custom KPIs Not Working
**Common Issues:**
- **DAX Syntax Errors** → Validate formula in Power BI Desktop
- **Relationship Issues** → Check table relationships in model
- **Performance Timeouts** → Optimize calculation or add aggregations
- **Data Type Mismatches** → Verify field types across tables

### 4. Alert & Notification Issues

#### Not Receiving Expected Alerts
**Diagnostic Steps:**
1. **Email Delivery:**
   ```
   ✓ Check spam/junk folders
   ✓ Verify email address in subscription
   ✓ Test with manual alert trigger
   ✓ Confirm corporate email not blocking
   ```

2. **Threshold Configuration:**
   ```
   ✓ Alert thresholds set appropriately
   ✓ Conditions actually met by data
   ✓ Alert frequency not suppressing notifications
   ✓ Time window settings correct
   ```

3. **System Status:**
   ```
   ✓ Power Automate flows running
   ✓ No email service outages
   ✓ Subscription still active
   ```

#### Alert Fatigue (Too Many Alerts)
**Optimization Steps:**
1. **Adjust Sensitivity:**
   - Increase percentage thresholds (e.g., 85% → 80%)
   - Extend time windows (daily → weekly)
   - Use digest format instead of individual alerts

2. **Smart Features:**
   - Enable intelligent suppression
   - Configure alert batching
   - Set business hours only delivery

3. **Team Configuration:**
   - Use shared team channels
   - Designate alert champions
   - Implement escalation tiers

### 5. Performance Issues

#### Dashboard Loading Slowly (>10 seconds)
**Immediate Actions:**
1. **Network Check:**
   - Test internet speed (min 10 Mbps recommended)
   - Close bandwidth-heavy applications
   - Try wired connection over WiFi

2. **Browser Optimization:**
   - Close unnecessary tabs (keep <10 open)
   - Disable browser extensions
   - Restart browser completely
   - Try different browser

3. **Dashboard Optimization:**
   - Use filtered views instead of full datasets
   - Switch to mobile layout for faster loading
   - Access during off-peak hours (avoid 9-11 AM)

**Performance Benchmarks:**
- **Acceptable:** <5 seconds initial load, <2 seconds interactions
- **Poor:** 5-15 seconds load time
- **Unacceptable:** >15 seconds or timeouts

#### Query Timeouts
**Resolution:**
1. Reduce scope: Shorter time periods, fewer capabilities
2. Use pre-built summary views when available
3. Schedule complex reports instead of real-time viewing
4. Contact admin for query optimization

---

## Advanced Issues

### Integration Problems

#### Confluence Sync Failures
**Error Indicators:**
- Configuration changes not reflected after 24 hours
- "Sync Failed" notifications
- Inconsistent SLO targets across capabilities

**Troubleshooting:**
1. **Verify Confluence Access:**
   - Service account can read SLO configuration pages
   - No recent permission changes
   - API endpoints responding

2. **Check Configuration Format:**
   - Tables properly formatted in Confluence
   - Required fields populated
   - No special characters causing parsing issues

3. **Manual Sync Options:**
   - Export configuration from Confluence
   - Import via admin interface
   - Restore from last known good configuration

#### Email System Integration
**Common Issues:**
- **SMTP Authentication:** Check service account credentials
- **Corporate Firewall:** Verify email server access
- **Message Blocking:** Add to safe senders list
- **Format Issues:** Test with simple text before HTML

### Mobile Access

#### Mobile App Problems
**Standard Resolution:**
1. Update to latest Power BI mobile app version
2. Log out/in to refresh authentication
3. Clear app cache and data
4. Reinstall app if issues persist

#### Mobile Performance
**Optimization:**
- Use mobile-specific dashboards
- Limit to key metrics only
- Switch to WiFi from cellular
- Close other mobile apps to free memory

---

## Self-Service Diagnostic Tools

### Built-in Validation Measures
Access these in the "System Health" dashboard:

1. **Relationship_Validation**: Checks data model integrity
2. **Data_Quality_Summary**: Identifies missing/incorrect data
3. **Performance_Test**: Measures calculation speed
4. **User_Access_Test**: Validates permission levels

### Quick Health Check
Run this 5-minute diagnostic before escalating:

```
1. Can I access main dashboard? [YES/NO]
2. Is data less than 48 hours old? [YES/NO]  
3. Do SLO percentages look reasonable? [YES/NO]
4. Are my subscriptions receiving emails? [YES/NO]
5. Does page load in under 10 seconds? [YES/NO]
```

If any answer is "NO", use relevant section above before escalating.

---

## Escalation & Support

### Contact Information

**L1 Support (General Issues)**
- **Email:** slo-support@company.com
- **Teams:** SLO Dashboard Support
- **Hours:** Business hours, 2-hour response

**L2 Support (Technical Issues)**  
- **Email:** platform-support@company.com
- **Emergency:** Extension 5555
- **Hours:** Extended hours, 1-hour critical response

**L3 Support (Development)**
- Accessed via L2 escalation only
- For customizations, integrations, architecture issues

### Escalation Information Package

**Always include:**
```
□ Issue description and error messages
□ Steps to reproduce 
□ Time issue started
□ Users/capabilities affected
□ Business impact assessment
□ Screenshots of error states
□ Browser/device information
□ Attempted resolution steps
```

### Emergency Procedures

**System-Wide Outage:**
1. **Immediate:** Call L2 emergency line (Ext 5555)
2. **Parallel:** Email L2 with "CRITICAL" in subject
3. **Communication:** Post in support Teams channel
4. **Stakeholders:** Notify capability owners if >2 hours

**Data Corruption Suspected:**
1. **Stop:** Cease all configuration changes immediately
2. **Document:** Screenshot affected areas
3. **Preserve:** Export current data if possible
4. **Escalate:** Direct to L3 via L2 urgent path

---

## Prevention & Maintenance

### Proactive Monitoring
**Weekly Checks:**
- Review refresh logs for failures
- Monitor performance trends
- Check alert delivery rates
- Validate random sample calculations

**Monthly Reviews:**
- User feedback analysis
- Configuration change impact assessment
- Performance optimization opportunities
- Documentation updates needed

### Best Practices for Users
1. **Regular Health Checks:** Use diagnostic tools monthly
2. **Configuration Discipline:** Test changes in non-production first
3. **Performance Awareness:** Monitor your dashboard response times
4. **Issue Reporting:** Document problems immediately while fresh
5. **Training Currency:** Attend quarterly refresh sessions

---

*Last Updated: [Current Date] | For immediate assistance with critical issues, contact L2 Support at platform-support@company.com*