# Power Query Analysis: SLO Dashboard System

## Introduction

This document provides a comprehensive analysis of the Service Level Objective (SLO) dashboard Power Query system. The system is designed to track and visualize service performance metrics across various capabilities, measuring key indicators like response time, lead time, and SLA achievement rates.

## Query Purposes

| Query Name | Purpose | Input | Output |
|------------|---------|-------|--------|
| **Fact_Ticket_Summary** | Creates the primary fact table for ticket analysis | Ticket data from Excel/CSV/SharePoint | Fact table with SLA metrics, resolution times, and capability information |
| **Fact_Status_Change** | Tracks ticket status transitions and timing | Status change history | Status change events with duration calculations and milestone flags |
| **Dim_Date** | Provides time-based analysis framework | Generated date range (2023-2027) | Comprehensive date dimension with calendar hierarchies and business day flags |
| **Dim_Capability** | Defines service capabilities and targets | SLO configuration data | Dimension table of service capabilities with SLA target times |
| **Dim_Status** | Standardizes ticket status categorization | Static status definitions | Status dimension with category mapping and time calculation flags |
| **Config_Issue_Type_Mapping** | Maps ticket types to capability areas | Mapping configuration | Configuration table linking issue types to capability keys |
| **Default_SLA_Table** | Provides fallback SLA targets | Static SLA definitions | Default response time targets by ticket type |
| **Business Hours Calculation** | Calculates working hours between timestamps | Start/end timestamps | Business hours elapsed, accounting for work hours and days |

## Data Flow Diagram

```
┌───────────────────┐            ┌────────────────────┐
│  Jira_Snapshot    │──reference→│ Fact_Ticket_Summary│
└───────────────────┘            └─────────┬──────────┘
                                           │
                                           │ joins/lookups
                                           ▼
┌───────────────────┐            ┌─────────────────────┐
│ Business Hours    │            │      Dim_Date       │
│ Calculation       │←───uses────┤  (date dimension)   │
│ Function          │            └─────────────────────┘
└───────┬───────────┘                      ▲
        │                                  │
        │                                  │
        ▼                                  │
┌───────────────────┐                      │
│ Fact_Status_Change│─────joins date─────→┘
└────────┬──────────┘
         │
         │ joins by key
         │
         ▼
┌───────────────────────────┐    ┌────────────────────┐
│Config_Issue_Type_Mapping  │←───┤  Dim_Capability    │
└──────────┬────────────────┘    └────────────────────┘
           │                               ▲
           │joins by issue_type            │
           ▼                               │
┌───────────────────┐                      │
│Fact_Ticket_Summary│─────joins by CapabilityKey───────┘
└─────────┬─────────┘
          │
          │ looks up SLA targets
          ▼
┌───────────────────┐            ┌────────────────────┐
│  Default_SLA_Table│            │     Dim_Status     │
└───────────────────┘            │ (status categories)│
                                 └────────────────────┘
```

## Key Issues Identified

### 1. Data Source Inconsistencies

**Issue:** Multiple data sources referenced across queries with planned but unimplemented transition to Jira data.

**Impact:** Creates maintenance burden and potential data synchronization issues.

### 2. Duplicated Business Logic

**Issue:** Critical business rules are implemented multiple ways:
- SLA target determination logic in both Power Query and DAX
- Business hour calculations in three different places
- Status categorization duplicated rather than referencing dimensions

**Impact:** Changes require updates in multiple places, increasing error risk.

### 3. Naming and Reference Inconsistencies

**Issue:** Inconsistent column naming between Power Query scripts and DAX measures.

**Impact:** Creates references to non-existent columns and makes code harder to follow.

### 4. Hardcoded Configuration Values

**Issue:** Business rules, status lists, and file paths hardcoded throughout scripts.

**Impact:** Configuration changes require code modifications rather than parameter updates.

### 5. Limited Error Handling

**Issue:** Minimal protection against missing data, schema changes, or calculation errors.

**Impact:** Pipeline fragility when data quality issues arise.

## Recommended Improvements

### For Business Users

1. **Centralized Configuration**
   - Create a single "control panel" for all adjustable business rules
   - Allow non-developers to update SLA targets and mappings without code changes

2. **Self-Documenting Reports**
   - Add calculation explanations within the dashboard
   - Include data freshness indicators and quality metrics

3. **Incremental Data Refresh**
   - Reduce refresh times by only loading new or changed tickets
   - Enable more frequent dashboard updates

### For Developers

1. **Standardize Naming and Structure**
   - Enforce consistent naming conventions
   - Create reusable function modules for shared logic

2. **Improve Data Validation**
   - Add explicit error checking for critical operations
   - Create diagnostic views to monitor data quality

3. **Optimize Query Performance**
   - Structure operations to support query folding
   - Buffer small lookup tables for performance
   - Move filtering operations earlier in processing chains

### Implementation Priority

| Improvement | Effort | Impact | Priority |
|-------------|--------|--------|----------|
| Jira Integration | Medium | High | 1 |
| Centralize Status Logic | Low | Medium | 2 |
| Extract Shared Functions | Medium | Medium | 3 |
| Standardize Naming | Low | Medium | 4 |
| Add Error Handling | Medium | High | 5 |

## Conclusion

The SLO Dashboard Power Query system effectively tracks key performance metrics, but suffers from structural issues that make maintenance challenging. By addressing the identified issues and implementing the recommended improvements, the system will become more robust, maintainable, and adaptable to changing business requirements.

The highest priority should be completing the Jira integration to establish a single authoritative data source, followed by centralizing the status logic to ensure consistent business rule application throughout the system.s