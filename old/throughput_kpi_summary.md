# Throughput KPI Model Summary

## Overview

This document summarizes the data model changes made to support the **Throughput KPI** in the Power BI SLO Dashboard. The Throughput KPI measures the count of completed tickets grouped by ticket type and completion month.

---

## 1. New Fields Added to Fact_Ticket_Summary

### `Is_Completed` (Boolean)
- **Purpose**: Flags tickets as completed based on their current status
- **Logic**: `TRUE` if status = "Done", "Closed", or "Resolved"; `FALSE` otherwise
- **Benefit**: Pre-calculated boolean for fast filtering in DAX measures

### `Completion_Date` (DateTime)
- **Purpose**: Stores the actual completion timestamp for each ticket
- **Logic**: 
  1. Uses `resolution_date` if available (primary source)
  2. Falls back to timestamp of last status change to Done/Closed/Resolved
  3. Returns `null` if ticket is not completed
- **Benefit**: Accurate completion tracking regardless of Jira field population

### `CompletedDate` (Date)
- **Purpose**: Date-only version of `Completion_Date` for relationship to Date dimension
- **Logic**: `Date.From(Completion_Date)` if not null, otherwise null
- **Benefit**: Enables calendar-based filtering and grouping

---

## 2. New Relationship Added

```
Fact_Ticket_Summary[CompletedDate] → Dim_Date[Date] (Many-to-One, Inactive)
```

### Key Details:
- **Inactive Relationship**: Must use `USERELATIONSHIP()` in DAX
- **Purpose**: Enables filtering/grouping by completion month/period
- **Supports**: Monthly aggregation, quarterly trends, year-over-year analysis

---

## 3. Throughput KPI Implementation

### DAX Measure
```dax
Throughput_KPI = 
CALCULATE(
    COUNTROWS(Fact_Ticket_Summary),
    Fact_Ticket_Summary[Is_Completed] = TRUE,
    USERELATIONSHIP(Fact_Ticket_Summary[CompletedDate], Dim_Date[Date])
)
```

### Formula Logic
- **Filter**: Only tickets where `Is_Completed = TRUE`
- **Date Context**: Uses `CompletedDate` via inactive relationship
- **Result**: Count of completed tickets by selected time period and ticket type

---

## 4. Global Filter Support

The Throughput KPI works seamlessly with standard dashboard filters:

| **Filter Type** | **Implementation** | **Effect** |
|-----------------|-------------------|------------|
| **Date** | `USERELATIONSHIP(CompletedDate, Dim_Date)` | Filters by completion period |
| **Ticket Type** | Direct filter on `issue_type` | Specific ticket types (Bug, Story, etc.) |
| **Capability** | Via `Config_Issue_Type_Capability_Mapping` | Business capability grouping |
| **Team** | Filter on `assignee_display_name` | Team/individual assignment |

### Filter Flow Example:
```
User selects "Data Quality" capability → 
Maps to issue types → 
Filters Fact_Ticket_Summary → 
Applies Throughput_KPI calculation
```

---

## 5. Performance Optimizations

### Design Benefits:
- **Pre-calculated `Is_Completed`**: No runtime status string comparisons
- **Efficient Date Relationship**: Leverages Power BI's optimized date tables
- **Boolean Storage**: Minimal memory footprint (~1 bit per row)
- **Monthly Aggregation**: Sub-second response times for typical datasets

### Technical Advantages:
- Calculations occur during data refresh, not query time
- Boolean and date filters pushed to storage engine
- Columnar compression optimizes boolean/date storage

---

## 6. Implementation Impact

### For Developers:
- New fields automatically available in fact table
- Standard DAX patterns for throughput analysis
- Maintains existing model structure and relationships

### For Business Users:
- Native support for monthly throughput reporting
- Consistent filtering across all dashboard elements
- Accurate completion tracking regardless of Jira configuration

### For Reports:
- Enable throughput trend analysis by month/quarter
- Compare completion rates across ticket types
- Track team/capability delivery velocity
- Support drill-down from summary to detail levels

---

## 7. Usage Examples

### Basic Throughput by Month
```dax
Monthly_Throughput = [Throughput_KPI]
// Use with Date table in visual
```

### Throughput by Capability
```dax
Capability_Throughput = 
CALCULATE(
    [Throughput_KPI],
    RELATED(Config_Issue_Type_Capability_Mapping[CapabilityKey]) = "DQ"
)
```

### Team Throughput Comparison
```dax
Team_Throughput = 
CALCULATE(
    [Throughput_KPI],
    Fact_Ticket_Summary[assignee_display_name] = "John Doe"
)
```

---

## Summary

The Throughput KPI implementation adds minimal complexity to the existing model while providing powerful analytical capabilities. The design prioritizes performance and usability, ensuring stakeholders can confidently analyze delivery velocity across multiple dimensions without technical barriers.

**Key Success Factors:**
- ✅ Accurate completion detection
- ✅ Flexible time-based analysis  
- ✅ Global filter compatibility
- ✅ Performance optimized
- ✅ Developer friendly implementation