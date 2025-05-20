# Power Query Dependency Map

Below is a structured dependency map showing the data flow between the Power Query scripts, including both explicit dependencies and logical relationships:

```
                          ┌───────────────────┐
                          │   Jira_Snapshot   │ (Mentioned in refactoring plan)
                          └─────────┬─────────┘
                                    │ (Planned reference)
                                    ▼
┌───────────────────┐     ┌───────────────────────┐
│  Dim_Capability   │◄────┤  Fact_Ticket_Summary  │
└───────────────────┘     └───────────┬───────────┘
        ▲                             │
        │                             │
        │                             │
┌───────┴───────────┐                │
│Default_SLA_Table  │◄───────────────┘
└───────────────────┘                │
        ▲                            │
        │                            │
        │                            │
┌───────┴───────────────┐           │
│Config_Issue_Type_     │◄──────────┘
│Capability_Mapping     │
└─────────────────────┬─┘
                      │
                      │
                      │
                      ▼
            ┌─────────────────────┐
            │  Fact_Status_Change │ (Implied relationship)
            └──────────┬──────────┘
                       │
                       │
                       ▼
               ┌───────────────┐
               │   Dim_Date    │ (Used in model relationships)
               └───────────────┘
                       ▲
                       │
                       │
              ┌────────┴────────┐
              │   Dim_Status    │ (Logically related to statuses)
              └─────────────────┘
```

## Data Flow Explanation

1. **Primary Data Sources**:
   * **Jira_Snapshot**: Mentioned in the refactoring plan, will become the primary data source for ticket information
   * **Fact_Ticket_Summary**: Currently loads directly from Excel/CSV, but will reference Jira_Snapshot after refactoring
   * **Fact_Status_Change**: Loads ticket status change history independently from Excel/CSV

2. **Reference Tables**:
   * **Dim_Capability**: Contains capability SLA targets referenced by Fact_Ticket_Summary
   * **Default_SLA_Table**: Provides fallback SLA values when capability-specific ones aren't available
   * **Config_Issue_Type_Capability_Mapping**: Maps issue types to capability keys
   * **Dim_Date**: Generated date dimension used in model relationships
   * **Dim_Status**: Contains status definitions and metadata

3. **Key Data Flows**:
   * **SLA Calculation Flow**: 
     Fact_Ticket_Summary → Config_Issue_Type_Capability_Mapping → Dim_Capability → Default_SLA_Table
     This implements the 2-tier SLA hierarchy logic, checking first for capability-level SLAs, then defaulting to standard SLAs

   * **Implied Ticket-Status Relationship**:
     Fact_Ticket_Summary → Fact_Status_Change
     While not explicitly coded as a query dependency, these tables have a logical relationship in the data model as they track the same tickets

   * **Time Intelligence**:
     Both Fact tables relate to Dim_Date for time-based analysis (as used in DAX measures)

This map shows both the current explicit dependencies and the planned refactoring, providing a complete picture of how data flows between the various Power Query scripts.