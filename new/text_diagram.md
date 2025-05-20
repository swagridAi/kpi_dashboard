# SLO Dashboard: From Raw Data to KPIs

## High-Level Data Flow Diagram

```
┌─────────────────┐     ┌────────────────────────┐     ┌────────────────────┐     ┌───────────────────┐
│  SOURCE SYSTEMS │     │  POWER QUERY EXTRACT   │     │ DATA TRANSFORMATION │     │    DATA MODEL     │
└────────┬────────┘     └───────────┬────────────┘     └─────────┬──────────┘     └─────────┬─────────┘
         │                          │                            │                          │
         ▼                          ▼                            ▼                          ▼
┌────────────────┐     ┌────────────────────────┐     ┌────────────────────┐     ┌───────────────────┐
│                │     │                        │     │                    │     │                   │
│  Jira Tickets  ├────►│   Jira_Snapshot Data   ├────►│ Fact_Ticket_Summary├────►│   KPI MEASURES    │
│                │     │                        │     │                    │     │                   │
└────────────────┘     └────────────────────────┘     └────────────────────┘     │  • Service Quality│
                                                                                  │  • Response Time  │
┌────────────────┐     ┌────────────────────────┐     ┌────────────────────┐     │  • Throughput     │
│                │     │                        │     │                    │     │  • Issue Resolution│
│ Jira Changelog ├────►│Status Change History   ├────►│Fact_Status_Change  ├────►│  • Lead Time      │
│                │     │                        │     │                    │     │  • Cycle Time      │
└────────────────┘     └────────────────────────┘     └────────────────────┘     │                   │
                                                                                  └───────────────────┘
┌────────────────┐     ┌────────────────────────┐     ┌────────────────────┐               │
│  Configuration │     │                        │     │                    │               │
│  Files         │     │  • Capability Mapping  │     │  • Dim_Capability  │               │
│  • SLA Targets ├────►│  • SLA Defaults       ├────►│  • Dim_Status      ├───────────────┘
│  • Mappings    │     │  • Status Categories   │     │  • Config Tables   │
└────────────────┘     └────────────────────────┘     └────────────────────┘

┌────────────────┐     ┌────────────────────────┐     ┌────────────────────┐     ┌───────────────────┐
│                │     │                        │     │                    │     │                   │
│   Generated    │     │                        │     │                    │     │                   │
│   Date Range   ├────►│    Date Generation     ├────►│     Dim_Date       ├────►│  Time Intelligence│
│                │     │                        │     │                    │     │                   │
└────────────────┘     └────────────────────────┘     └────────────────────┘     └───────────────────┘
```

## Key Data Transformation Steps

1. **Extract Phase**
   * Jira ticket data → `Jira_Snapshot`
   * Status change history → Status change records
   * Configuration files → Capability mappings and SLA targets

2. **Transform Phase**
   * `Jira_Snapshot` → `Fact_Ticket_Summary` with SLA flags and calculations
   * Status changes → `Fact_Status_Change` with business hours and workflow markers
   * Reference data → Dimension tables with business categorizations

3. **Model Phase**
   * Establish relationships between fact and dimension tables
   * Apply DAX measures to calculate KPIs
   * Create time intelligence calculations

4. **KPI Output**
   * Lead Time: Backlog-to-work transition time
   * Cycle Time: Active work duration
   * Response Time: End-to-end resolution time
   * Throughput: Completed tickets per period
   * Service Quality: SLA target achievement percentage
   * Issue Resolution Time: Average time to resolution

## Business Value Flow

```
Raw Data → Cleaned Data → Applied Business Rules → Calculated Metrics → Actionable Insights
  │            │               │                      │                    │
  │            │               │                      │                    │
  ▼            ▼               ▼                      ▼                    ▼
Extraction → Validation → Business Logic → Performance Measurement → Process Improvement
```

This data pipeline transforms raw ticket data into meaningful service level metrics that enable process improvement, capacity planning, and enhanced customer satisfaction through reliable service delivery.