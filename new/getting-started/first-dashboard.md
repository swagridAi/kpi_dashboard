# Understanding Your First Dashboard

Congratulations! You've completed the initial setup and can now see your team's performance data. This guide will help you understand what you're looking at and how to make the most of your first week with the SLO dashboard.

## What You're Looking At

Your dashboard displays three main sections organized for quick comprehension:

### Executive Summary Cards (Top Row)
- **Overall SLO Achievement Rate**: Large percentage showing tickets resolved within target timeframes (goal: 95%)
- **Month-over-Month Change**: Arrow and percentage indicating performance trend from last month
- **Six-Month Average**: Rolling average that smooths out monthly variations

### Performance Trend Charts (Center)
- **SLO Performance Line Chart**: Shows your achievement rate over the last six months
  - Red horizontal line = 95% target
  - Green line with data points = actual monthly performance
  - Data labels show exact percentages for each month
- **Volume Chart**: Displays ticket throughput (completed tickets per month)

### Detailed Breakdown (Bottom)
- **By Service Type**: Performance split across your services (Data Quality, Data Extracts, etc.)
- **Time-Based Metrics**: Lead Time, Cycle Time, and Response Time averages
- **At-Risk Tickets**: Current tickets approaching SLO deadlines

## Interpreting Initial Data

### If You See Limited Data

**This is completely normal for new capabilities!** You may notice:
- Partial months showing only recent weeks
- Single data points instead of trend lines  
- Some metrics showing "Insufficient Data"

**Timeline expectations:**
- **Week 1-2**: Basic current metrics appear, historical trends are sparse
- **Week 3-4**: Monthly patterns become visible, variance decreases
- **Month 2+**: Reliable trending data for strategic decisions

### If You See Surprising Numbers

**SLO achievement much lower than expected (below 80%)?**
- Verify your SLO targets align with actual service commitments
- Check if all relevant ticket types are included in calculations
- Consider whether targets reflect current process capabilities

**SLO achievement much higher than expected (above 98%)?**
- Targets might be too generous for meaningful performance tracking
- Some ticket categories might not be captured yet
- Historical performance may not reflect current workload complexity

**Tickets missing or strange volumes?**
- Ensure your team's Jira issues are assigned to the correct capability
- Verify status mappings match your actual workflow
- Check time filters match your expected analysis period

## Key Elements to Focus On

During your first week, prioritize understanding these core indicators:

### Current SLO Performance (Headlines)
- **Achievement rate**: Is it above 85% (acceptable) or below (needs attention)?
- **Service breakdown**: Which specific services are exceeding or missing targets?
- **Trend arrows**: Green ↗ (improving) vs. Red ↘ (declining) indicators

### Pattern Recognition
- **Daily/weekly cycles**: Do certain days consistently show better performance?
- **Ticket type variations**: Are bugs handled faster than feature requests?
- **Volume correlations**: Does higher volume correlate with longer resolution times?

### Actionable Insights
- **At-Risk tickets**: Current tickets likely to breach SLO without intervention
- **Recent completions**: Yesterday's resolved tickets and their resolution times
- **Bottleneck indicators**: Services showing consistently longer cycle times

## Common First-Time Questions

### Why is my SLO percentage low/high?

**For low percentages (below 85%):**
Your dashboard might be revealing previously hidden process inefficiencies. Check if:
- SLO targets reflect realistic service commitments to customers
- All workflow bottlenecks are accounted for in your time calculations
- Historical tickets included unusual circumstances (holidays, staff changes)

**For high percentages (above 95%):**
Consider whether your metrics capture the full service experience:
- Are all customer-facing ticket types included?
- Do targets challenge your team to maintain excellence?
- Are you measuring the most impactful aspects of service delivery?

### How long until I have meaningful trends?

- **Immediate (Day 1)**: Current performance snapshot, obvious bottlenecks
- **Short-term (2-3 weeks)**: Pattern identification, process validation  
- **Full analysis (2-3 months)**: Reliable month-over-month comparisons, seasonal awareness

### What should I do if data looks wrong?

1. **Verify configuration first:**
   - Review your Confluence page issue type mappings
   - Confirm workflow status definitions match actual processes
   - Check that SLO targets match service level agreements

2. **Spot-check with manual calculations:**
   - Select 3-5 recent tickets and manually calculate resolution times
   - Compare results with dashboard figures
   - Pay attention to business day vs. calendar day differences

3. **Report persistent discrepancies:**
   - Document specific examples of incorrect calculations
   - Contact support with ticket IDs and expected vs. actual values
   - Include screenshots of unexpected dashboard results

## Your First Week Checklist

Use this checklist to establish productive dashboard habits:

### Daily Actions (5 minutes each morning)
- [ ] **Check overnight resolutions**: Review tickets completed since yesterday
- [ ] **Monitor at-risk tickets**: Identify tickets approaching SLO deadlines
- [ ] **Scan for anomalies**: Look for unusual patterns or outlier resolution times

### Configuration Validation (One-time, 30 minutes total)
- [ ] **Ticket categorization**: Verify all team tickets appear under correct capability
- [ ] **Issue type mapping**: Ensure bugs, stories, tasks map to appropriate services  
- [ ] **Status workflow**: Confirm lead/cycle/response time boundaries match reality

### Alert Setup (One-time, 15 minutes)
- [ ] **SLO breach notifications**: Configure email alerts for missed targets
- [ ] **Approaching deadline warnings**: Set up "at risk" ticket alerts
- [ ] **Test notification delivery**: Send yourself a sample alert

### Team Engagement (End of week)
- [ ] **Schedule team review**: 15-minute discussion of initial findings
- [ ] **Gather validation feedback**: Ask team if metrics match their experience
- [ ] **Document questions**: Note any confusion for follow-up with support

## Ready for More?

Once you're comfortable reading your dashboard:

- **Enhance daily operations**: Learn advanced filtering and analysis techniques in [Team Operations Guide](../dashboard-usage/team-operations.md)
- **Customize your experience**: Adjust SLO targets and alert preferences in [Basic Configuration](basic-configuration.md)
- **Share insights proactively**: Use dashboard data to demonstrate value and identify improvement opportunities

Remember: This dashboard reveals your current state to help you improve, not to judge past performance. Focus on trends and actionable insights rather than absolute numbers, especially during your first month.

---

**Questions or issues?** Contact the Centralized Data Team via the Service Portal or join our weekly office hours every Tuesday at 2 PM.