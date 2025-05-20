# SLO Dashboard: Understanding the Service Performance Tracking System

## What Is This System?

The Service Level Objective (SLO) Dashboard is a performance tracking system built to monitor how well teams are handling service requests and tickets. Think of it as a "fitness tracker" for IT and service teams that shows:

- How quickly teams respond to requests
- How efficiently they resolve issues
- Whether they're meeting their promised service targets
- Trends in workload and performance over time

This system pulls data from Jira (a ticket tracking tool) and transforms it into meaningful metrics that help both team members and managers understand service performance.

## Why This Matters

Without a system like this, it's difficult to:
- Know if teams are meeting their service commitments
- Identify bottlenecks or delays in service delivery
- Recognize trends (improving or declining performance)
- Make data-driven decisions about resource allocation

## The 6 Core Metrics Being Tracked

The dashboard focuses on six essential measurements:

1. **Lead Time** (Days from ticket creation until work begins)
   - *How long do requests sit before someone starts working on them?*

2. **Cycle Time** (Days from work start to completion)
   - *Once started, how long does it take to finish the work?*

3. **Response Time** (Total days from request to resolution)
   - *From the customer's perspective, how long until their issue is resolved?*

4. **Throughput** (Number of tickets completed per period)
   - *How many requests can the team handle in a given timeframe?*

5. **Service Quality** (Percentage of tickets meeting SLA targets)
   - *How often are service level agreements being met?*

6. **Issue Resolution Time** (Average days to resolve issues)
   - *Typically, how long does resolution take?*

## How the Data is Organized

The system is built on several interconnected data tables:

### Main Data Tables

1. **Ticket Summary** - The central record of all service tickets
   - Contains core information: ticket ID, type, status, dates, assignee
   - Tracks whether each ticket met its service target
   - Calculates how long each ticket took to resolve

2. **Status Changes** - History of ticket progress
   - Records each time a ticket changes status
   - Measures how long tickets spend in each phase
   - Identifies key transitions (like when work begins)

### Supporting Reference Tables

3. **Calendar** - A complete date table
   - Identifies business days vs. weekends/holidays
   - Provides time period breakdowns (weeks, months, quarters)
   - Enables consistent time-based analysis

4. **Capability Categories** - Service type groupings
   - Defines different service capabilities (Data Quality, Engineering, etc.)
   - Sets specific target resolution times for each capability
   - Assigns ownership and maturity levels

5. **Status Definitions** - What each status means
   - Defines which statuses count for different time measurements
   - Groups statuses into categories (To Do, In Progress, Done, etc.)

6. **Configuration Tables** - System settings
   - Maps ticket types to capability categories
   - Provides default service targets when specific ones aren't defined

## How It Works: The Data Journey

1. **Data Collection**
   - Ticket data is extracted from Jira into Excel files
   - Files are stored at standard locations (like `C:\SLOData\ticket_summary.xlsx`)
   - Alternative sources include SharePoint lists or direct database connections

2. **Data Transformation**
   - The system loads and cleans the raw ticket data
   - It calculates time metrics (days to resolve, business hours spent)
   - It determines if service targets were met for each ticket
   - It flags tickets at risk of missing targets

3. **Metric Calculation**
   - The clean data feeds into calculations (DAX measures)
   - These calculate the 6 core metrics plus variations:
     - Monthly, quarterly, and yearly metrics
     - Trending and comparative metrics
     - Team and capability-specific metrics

4. **Visualization in Dashboard**
   - The calculated metrics appear in the final dashboard
   - Users can filter by time period, team, or service category
   - Performance summaries show status at a glance

## Business Rules and Logic

The system incorporates specific business rules:

- **Service Level Targets** follow a 2-tier hierarchy:
  1. First, look up the target based on the capability category
  2. If not found, fall back to a default target based on ticket type
  3. If still not found, use 5 days as the ultimate default

- **Business Hours Calculation**:
  - Standard business hours are 9 AM to 5 PM, Monday to Friday
  - Weekends and holidays are excluded from business hour calculations

- **Status Transitions**:
  - Lead time starts when a ticket moves from "Backlog" to "In Progress"
  - Cycle time measures only the active work phase
  - A ticket is considered complete when it reaches statuses like "Done" or "Closed"

## Planned Improvements

According to the documentation, there's a plan to:
- Replace the standalone Excel file sources with direct references to Jira data
- Consolidate data loading to reduce duplication and maintenance
- Eliminate local file dependencies

## Summary

The SLO Dashboard transforms raw ticket data into meaningful performance metrics that help teams track their service quality. By focusing on six core metrics across different time periods, it provides a complete picture of service performance that can drive improvements and ensure customer satisfaction.