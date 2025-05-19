# Issue Resolution KPI Summary

## Overview

This document explains two key performance indicators (KPIs) that help us understand how effectively we're resolving issues: **Average Time to Resolution** and **Reopened Rate**. These metrics provide valuable insights into our service quality and efficiency.

## Average Time to Resolution

### What it measures
Average Time to Resolution tells us how long, on average, it takes for a ticket to go from creation to final resolution, measured in days. This KPI only includes tickets that have been completed.

### Why it's important
This metric directly reflects our ability to resolve issues in a timely manner. Lower numbers generally indicate better service for our customers. It helps us:
- Evaluate if we're meeting our service level objectives (SLOs)
- Identify trends in resolution efficiency
- Compare performance across different teams and capabilities
- Set realistic expectations for stakeholders

### How it's calculated
The system calculates this by:
1. Looking at all completed tickets
2. Measuring the days between when each ticket was created and when it was resolved
3. Calculating the average of those durations

### Recommended visualization
- **KPI card** showing the current average with an icon indicating whether it's improving or worsening
- **Line chart** showing the trend over time (last 6 months)
- **Bar chart** comparing average resolution times across different capabilities or teams
- Consider adding a **reference line** showing the target resolution time

## Reopened Rate

### What it measures
Reopened Rate shows the percentage of tickets that were reopened after being marked as resolved or done. A "reopened" ticket is one that moved from a completed state (like "Done" or "Resolved") back to an active state (like "Open" or "In Progress").

### Why it's important
This metric highlights potential quality issues in our resolution process. When tickets are reopened, it usually means:
- The initial solution didn't fully address the problem
- The issue was closed prematurely
- New related problems emerged
- There was a misunderstanding about requirements

A lower reopened rate generally indicates higher quality work and better communication.

### How it's calculated
The system calculates this by:
1. Identifying tickets that had a status change from a completed state back to an active state
2. Counting how many tickets were reopened
3. Dividing by the total number of tickets
4. Converting to a percentage

### Recommended visualization
- **KPI card** showing the current reopened rate with a visual indicator of performance
- **Line chart** tracking the reopened rate over time
- **Heat map** showing reopened rates by capability and issue type to identify problem areas
- **Gauge visual** comparing current performance to targets

## Dashboard Integration

### Placement
These KPIs should be featured on:
- The executive summary page for a high-level overview
- The service quality section for more detailed analysis
- Capability-specific pages for team-level insights

### Context
Always display these metrics with appropriate context:
- Trend indicators (improving/worsening)
- Comparisons to targets
- Historical averages
- Relevant filters (capability, issue type, priority)

## Maintenance

### Target Setting
- Review and adjust targets quarterly
- Set targets based on historical performance and business requirements
- Different capabilities may have different targets based on complexity

### Data Quality
For accurate measurements:
- Ensure tickets are properly closed when the work is complete
- Train teams on proper status workflow management
- Review and update status transition rules as needed

### Regular Review
- Review these metrics in monthly service performance meetings
- Investigate significant changes in either metric
- Use the insights to drive improvement initiatives

## Related Metrics

These KPIs work well when analyzed alongside:
- SLO Achievement Rate
- Customer Satisfaction Scores
- First-Time Resolution Rate
- Issue Complexity (as measured by status transitions)

By monitoring these metrics together, you'll gain a comprehensive understanding of service performance and can make informed decisions for improvement.
