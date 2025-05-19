# SLA KPI Model Summary

## Overview

We have enhanced the Power BI model to provide comprehensive Service Level Agreement (SLA) tracking for all tickets. This document explains the key updates in simple terms and how they will benefit your team.

## What We Added

### 1. Default SLA Reference Table

**What it is**: A lookup table that defines standard SLA targets for different types of tickets.

**Why we need it**: Every ticket needs an SLA target to measure performance against. Without this table, some tickets would have no SLA definition, making it impossible to track performance consistently.

**What's included**:
- **Bug tickets**: 3 days (critical issues need fast resolution)
- **Regular tasks**: 5 days (standard work items)
- **Large projects (Epics)**: 10 days (complex work needs more time)
- **Stories**: 7 days (feature development)
- **Small sub-tasks**: 2 days (quick components)
- **Incidents**: 1 day (urgent system problems)

**How it works**: Think of this as a "backup plan" - if a specific team hasn't set their own SLA targets, the system automatically uses these standard values so every ticket still has a target to measure against.

### 2. Smart SLA Assignment System

We've created a three-level system that automatically finds the right SLA for each ticket:

1. **First choice**: Use the specific SLA target set by the capability team (like Data Quality or Data Extracts)
2. **Second choice**: If no specific target exists, use the default SLA for that ticket type (from our reference table above)
3. **Last resort**: If all else fails, use 5 days as a basic fallback

**Example**: A bug reported to the Data Quality team would first check if Data Quality has set their own SLA for bugs. If they haven't, it would use the standard 3-day SLA for bugs from our reference table.

### 3. Automatic SLA Performance Tracking

**What we added**: A new column called "Met_SLA" that automatically calculates whether each ticket was resolved within its SLA target.

**How it works**:
- For **resolved tickets**: Shows TRUE if completed on time, FALSE if it took too long
- For **open tickets**: Shows as blank (since they're not finished yet)
- **Calculation**: Takes the number of days from ticket creation to resolution and compares it to the SLA target

**Business value**: No more manual checking or calculations - you can instantly see which tickets met their SLA and which didn't.

## What This Means for Your Dashboard

### 1. Executive Dashboard - SLA Overview

**New metrics you'll see**:
- **Overall SLA Performance**: Percentage of tickets resolved within SLA (goal: 95%)
- **SLA Trend**: How performance is changing month over month
- **6-Month Average**: Rolling average to smooth out monthly variations

**Visual examples**:
- Green line showing actual performance vs. red line showing the 95% target
- Clear indicators when performance drops below acceptable levels

### 2. Capability Team Dashboards

**For Data Quality, Data Extracts, Change Controls, etc.**:
- **Team-specific SLA performance**: How well each capability is meeting their targets
- **Comparison views**: See how your team performs against others
- **Breakdown by ticket type**: Understand which types of work are meeting SLA and which aren't

### 3. Operational Dashboards

**For day-to-day management**:
- **Tickets at risk**: Early warning for tickets approaching their SLA deadline
- **SLA breach analysis**: Which types of tickets are most likely to miss their SLA
- **Time remaining**: How much time is left before SLA deadline for open tickets

## Key Benefits

### 1. Complete Coverage
- **Before**: Some tickets had no SLA targets, making performance measurement inconsistent
- **After**: Every single ticket has an appropriate SLA target, ensuring complete performance visibility

### 2. Automatic Updates
- **Before**: Manual tracking and calculations were needed
- **After**: Everything updates automatically when new tickets are created or resolved

### 3. Flexible Configuration
- Teams can set their own specific SLA targets when ready
- Default values provide immediate coverage for new teams or ticket types
- System gracefully handles any gaps in configuration

### 4. Business Intelligence
- Clear visibility into what's working and what isn't
- Data-driven decisions about resource allocation
- Early warning system for potential SLA breaches

## How the System Handles Different Scenarios

### Scenario 1: Well-Configured Team
- **Data Quality team** has set specific SLA targets for bugs (2 days) and tasks (4 days)
- System uses these custom values for all Data Quality tickets
- Performance measured against team-specific targets

### Scenario 2: New Team
- **Records Management team** just joined but hasn't set SLA targets yet
- System automatically uses default SLA values (bugs: 3 days, tasks: 5 days)
- Team can start tracking performance immediately while they develop custom targets

### Scenario 3: Unusual Ticket Type
- Someone creates a "Research" ticket type that doesn't exist in our defaults
- System applies the 5-day fallback rule
- Admins get notified to add appropriate SLA definition

## Getting Started

### For Capability Owners
1. Review the default SLA values for your ticket types
2. Decide if you want to customize any SLA targets for your team
3. Work with the Data Team to configure your specific targets if desired

### For Executives
1. New SLA metrics will appear in your monthly reports automatically
2. Look for the 95% SLA achievement target line on charts
3. Focus on capabilities consistently below 90% for attention

### For Team Members
1. SLA indicators will appear on individual tickets
2. Filter views to see only tickets at risk of missing SLA
3. Use the data to identify process improvement opportunities

## Next Steps

1. **Week 1**: Default SLA table is implemented and calculating
2. **Week 2**: Dashboard updates go live with new SLA metrics
3. **Week 3**: Training sessions for capability owners on interpreting SLA data
4. **Week 4**: Begin monthly SLA performance reviews

## Questions or Concerns?

If you have questions about:
- **How SLA targets are assigned**: Contact the Data Governance team
- **Setting custom SLA targets**: Reach out to your capability lead
- **Dashboard access or training**: Schedule time with the Business Intelligence team

Remember: The goal is better service delivery through clear, measurable targets. This system provides the foundation for data-driven improvements in how we serve our internal customers.