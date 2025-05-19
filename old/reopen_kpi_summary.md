# Reopened Tickets KPI Implementation Summary

## Executive Overview

This document summarizes the comprehensive implementation of reopened ticket detection and tracking within the SLO Dashboard Power BI model. The implementation enables accurate measurement of issue resolution quality by distinguishing between tickets that were resolved correctly on the first attempt versus those requiring additional work after initial closure.

---

## 1. Implementation Architecture

### 1.1 Detection Logic in Fact_Ticket_Status_Change

**Purpose**: Identify individual status transitions that represent ticket reopening events

**Primary Field Added**: `ReopenEvent` (Boolean)

#### Power Query Implementation
```m
// Detect reopened tickets - transitions from Done-like to Open-like statuses
AddReopenEvent = Table.AddColumn(AddDurationBusiness, "ReopenEvent", each
    let
        // Enhanced reopening criteria with additional business rules
        DoneStatuses = {"Done", "Resolved", "Closed", "Completed", "Fixed", "Won't Fix", "Duplicate", "Cannot Reproduce"},
        OpenStatuses = {"To Do", "Open", "In Progress", "Backlog", "New", "Reopened", "Under Investigation", "In Review"},
        
        // Check basic status transition
        FromDoneStatus = List.Contains(DoneStatuses, [from_string]),
        ToOpenStatus = List.Contains(OpenStatuses, [to_string]),
        
        // Additional criteria for more precise reopen detection
        // Exclude transitions that are not true reopens (e.g., clerical corrections)
        NotImmediateCorrection = Duration.TotalMinutes([change_created] - [PreviousChangeTime]) > 30,
        
        // Check if the ticket had been resolved for a meaningful period
        WasActuallyResolved = Duration.TotalHours([change_created] - [PreviousChangeTime]) > 1,
        
        // Final evaluation
        IsReopenEvent = FromDoneStatus and ToOpenStatus and NotImmediateCorrection and WasActuallyResolved
    in
        IsReopenEvent)
```

#### Business Rules Applied
1. **Status Transition Validation**: Must transition from Done-like status to Open-like status
2. **Minimum Resolution Time**: Ticket must have been resolved for at least 1 hour
3. **Immediate Correction Filter**: Excludes changes made within 30 minutes (administrative corrections)
4. **Temporal Validation**: Ensures genuine reopening, not data corrections

### 1.2 Related Fields in Fact_Ticket_Status_Change

| Field Name | Data Type | Purpose |
|------------|-----------|---------|
| `ReopenEvent` | Boolean | True if status change represents a reopening |
| `IsReopenedTicket` | Boolean | Legacy field for backwards compatibility |
| `ReopeningType` | Text | Classification of reopening (Explicit, Review, Investigation, General) |
| `ReopenContext` | Text | JSON metadata about the reopen event |

---

## 2. Aggregation Logic in Fact_Ticket_Summary

### 2.1 Primary Aggregation Field

**Primary Field Added**: `Was_Reopened` (Boolean)

#### Power Query Implementation with Null Handling
```m
// COMPLETE WAS_REOPENED CALCULATION WITH NULL HANDLING
AddWasReopened = Table.AddColumn(AddLeadTime, "Was_Reopened", each
    let
        StatusChanges = [StatusChanges],
        
        // Handle null/missing changelog data - return false if no changes exist
        HasChanges = StatusChanges <> null and StatusChanges <> #table({}, {}),
        
        // If no status changes exist, ticket cannot be reopened
        Result = if not HasChanges then
            false
        else
            let
                // Check if ReopenEvent column exists in the changelog data
                HasReopenEventColumn = Table.HasColumns(StatusChanges, {"ReopenEvent"}),
                
                // If ReopenEvent column doesn't exist, fall back to legacy logic
                ReopenCheck = if not HasReopenEventColumn then
                    // Legacy reopening detection logic
                    let
                        // Define Done-like and Open-like statuses
                        DoneStatuses = {"Done", "Resolved", "Closed", "Completed", "Fixed", "Won't Fix", "Duplicate", "Cannot Reproduce"},
                        OpenStatuses = {"To Do", "Open", "In Progress", "Backlog", "New", "Reopened", "Under Investigation", "In Review"},
                        
                        // Find transitions from Done to Open statuses
                        ReopenTransitions = Table.SelectRows(StatusChanges, each
                            // Ensure both from_string and to_string are not null
                            [from_string] <> null and [to_string] <> null and
                            // Check for Done to Open transition
                            List.Contains(DoneStatuses, [from_string]) and
                            List.Contains(OpenStatuses, [to_string])
                        ),
                        
                        HasReopenTransitions = Table.RowCount(ReopenTransitions) > 0
                    in
                        HasReopenTransitions
                else
                    // Use ReopenEvent column if available
                    let
                        // Filter for valid ReopenEvent records (not null and true)
                        ReopenEvents = Table.SelectRows(StatusChanges, each
                            [ReopenEvent] <> null and [ReopenEvent] = true
                        ),
                        
                        HasReopenEvents = Table.RowCount(ReopenEvents) > 0
                    in
                        HasReopenEvents
            in
                ReopenCheck
    in
        Result)
```

### 2.2 Supporting Fields in Fact_Ticket_Summary

| Field Name | Data Type | Purpose |
|------------|-----------|---------|
| `Was_Reopened` | Boolean | True if ticket was ever reopened |
| `Reopen_Count` | Integer | Number of times ticket was reopened |
| `First_Reopen_Date` | Date | Date of first reopening event |
| `Days_To_First_Reopen` | Number | Days from resolution to first reopen |

#### Additional Field Calculations
```m
// Additional reopening metrics with null handling
AddReopenCount = Table.AddColumn(AddWasReopened, "Reopen_Count", each
    let
        StatusChanges = [StatusChanges],
        
        // Return 0 if no changes exist
        Result = if StatusChanges = null or StatusChanges = #table({}, {}) then
            0
        else
            let
                // Check if ReopenEvent column exists
                HasReopenEventColumn = Table.HasColumns(StatusChanges, {"ReopenEvent"}),
                
                Count = if not HasReopenEventColumn then
                    // Legacy count logic
                    let
                        DoneStatuses = {"Done", "Resolved", "Closed", "Completed", "Fixed", "Won't Fix", "Duplicate", "Cannot Reproduce"},
                        OpenStatuses = {"To Do", "Open", "In Progress", "Backlog", "New", "Reopened", "Under Investigation", "In Review"},
                        
                        ReopenTransitions = Table.SelectRows(StatusChanges, each
                            [from_string] <> null and [to_string] <> null and
                            List.Contains(DoneStatuses, [from_string]) and
                            List.Contains(OpenStatuses, [to_string])
                        ),
                        
                        ReopenCount = Table.RowCount(ReopenTransitions)
                    in
                        ReopenCount
                else
                    // Count ReopenEvent = true records
                    let
                        ReopenEvents = Table.SelectRows(StatusChanges, each
                            [ReopenEvent] <> null and [ReopenEvent] = true
                        ),
                        
                        ReopenCount = Table.RowCount(ReopenEvents)
                    in
                        ReopenCount
            in
                Count
    in
        Result)
```

---

## 3. Alternative DAX Implementation

### 3.1 DAX Calculated Column
```dax
// Was_Reopened calculated column with complete null handling
Was_Reopened = 
VAR TicketKey = Fact_Ticket_Summary[key]

// Check if any status changes exist for this ticket
VAR HasStatusChanges = 
    NOT ISBLANK(
        CALCULATE(
            COUNTROWS(Fact_Ticket_Status_Change),
            Fact_Ticket_Status_Change[TicketKey] = TicketKey
        )
    )

// If no status changes, return false
VAR Result = 
    IF(
        NOT HasStatusChanges,
        FALSE,
        // Check for ReopenEvent column existence and use it
        VAR HasReopenEvents = 
            CALCULATE(
                COUNTROWS(Fact_Ticket_Status_Change),
                Fact_Ticket_Status_Change[TicketKey] = TicketKey,
                NOT ISBLANK(Fact_Ticket_Status_Change[ReopenEvent]),
                Fact_Ticket_Status_Change[ReopenEvent] = TRUE
            )
        
        // Fallback to legacy reopening detection if ReopenEvent doesn't exist
        VAR HasLegacyReopenings = 
            IF(
                HasReopenEvents > 0,
                TRUE,
                // Legacy detection logic
                VAR DoneStatuses = {"Done", "Resolved", "Closed", "Completed", "Fixed", "Won't Fix", "Duplicate", "Cannot Reproduce"}
                VAR OpenStatuses = {"To Do", "Open", "In Progress", "Backlog", "New", "Reopened", "Under Investigation", "In Review"}
                
                VAR LegacyReopenCount = 
                    CALCULATE(
                        COUNTROWS(Fact_Ticket_Status_Change),
                        Fact_Ticket_Status_Change[TicketKey] = TicketKey,
                        NOT ISBLANK(Fact_Ticket_Status_Change[from_string]),
                        NOT ISBLANK(Fact_Ticket_Status_Change[to_string]),
                        Fact_Ticket_Status_Change[from_string] IN DoneStatuses,
                        Fact_Ticket_Status_Change[to_string] IN OpenStatuses
                    )
                
                RETURN LegacyReopenCount > 0
            )
            
        RETURN HasLegacyReopenings
    )

RETURN Result
```

---

## 4. Data Model Relationships

### 4.1 Critical Relationship
**Fact_Ticket_Status_Change[TicketKey] → Fact_Ticket_Summary[key]** (Many-to-One)

- **Cardinality**: Many-to-One
- **Cross Filter Direction**: Both
- **Active**: Yes
- **Purpose**: Enables aggregation of reopen events from changelog to ticket level

### 4.2 Validation of Relationship
```dax
// Relationship validation measure
Relationship_Validation = 
VAR ChangesWithoutTickets = 
    CALCULATE(
        COUNTROWS(Fact_Ticket_Status_Change),
        ISBLANK(RELATED(Fact_Ticket_Summary[key]))
    )
VAR TicketsWithoutChanges = 
    CALCULATE(
        COUNTROWS(Fact_Ticket_Summary),
        ISBLANK(RELATED(Fact_Ticket_Status_Change[TicketKey]))
    )
RETURN
    "Orphaned Changes: " & ChangesWithoutTickets & 
    ", Tickets Without Changes: " & TicketsWithoutChanges
```

---

## 5. Validation Framework

### 5.1 Test Case Validation
```dax
// Known test case validation
Test_Case_Simple_Reopen = 
VAR TestTickets = 
    FILTER(
        Fact_Ticket_Summary,
        Fact_Ticket_Summary[key] IN {
            "TEST-001", // Known reopened ticket
            "TEST-002", // Known non-reopened ticket  
            "TEST-003"  // Multiple reopenings
        }
    )

VAR ValidationResults = 
    ADDCOLUMNS(
        TestTickets,
        "Expected_Was_Reopened", 
            SWITCH(
                [key],
                "TEST-001", TRUE,
                "TEST-002", FALSE,
                "TEST-003", TRUE,
                BLANK()
            ),
        "Expected_Reopen_Count",
            SWITCH(
                [key],
                "TEST-001", 1,
                "TEST-002", 0,
                "TEST-003", 3,
                BLANK()
            ),
        "Test_Result_Was_Reopened", 
            [Was_Reopened] = [Expected_Was_Reopened],
        "Test_Result_Reopen_Count",
            [Reopen_Count] = [Expected_Reopen_Count]
    )

RETURN ValidationResults
```

### 5.2 Statistical Validation
```dax
// Statistical validation with baseline comparison
Statistical_Validation_Summary = 
VAR CurrentMetrics = 
    SUMMARIZE(
        Fact_Ticket_Summary,
        "Total_Tickets", COUNTROWS(Fact_Ticket_Summary),
        "Reopened_Tickets", COUNTROWS(FILTER(Fact_Ticket_Summary, [Was_Reopened] = TRUE)),
        "Reopening_Rate", 
            DIVIDE(
                COUNTROWS(FILTER(Fact_Ticket_Summary, [Was_Reopened] = TRUE)),
                COUNTROWS(Fact_Ticket_Summary)
            ) * 100
    )

VAR HistoricalBaseline = 
    CALCULATE(
        DIVIDE(
            COUNTROWS(FILTER(Fact_Ticket_Summary, [Was_Reopened] = TRUE)),
            COUNTROWS(Fact_Ticket_Summary)
        ) * 100,
        DATESINPERIOD(Dim_Date[Date], MAX(Dim_Date[Date]), -6, MONTH)
    )

VAR ValidationFlags = 
    ADDCOLUMNS(
        CurrentMetrics,
        "Historical_Baseline", HistoricalBaseline,
        "Deviation_From_Baseline", [Reopening_Rate] - HistoricalBaseline,
        "Validation_Status",
            IF(
                ABS([Reopening_Rate] - HistoricalBaseline) > 5,
                "REVIEW_REQUIRED",
                IF(
                    ABS([Reopening_Rate] - HistoricalBaseline) > 2,
                    "MONITOR",
                    "NORMAL"
                )
            )
    )

RETURN ValidationFlags
```

### 5.3 Data Quality Checks
```m
// Power Query validation for data integrity
AddValidationFlags = Table.AddColumn(TypedTable, "Validation_Warnings", each
    let
        Warnings = {},
        
        // Check if Was_Reopened and Reopen_Count are consistent
        Warnings = if [Was_Reopened] = true and [Reopen_Count] = 0 then
            Warnings & {"Was_Reopened=true but Reopen_Count=0"}
        else if [Was_Reopened] = false and [Reopen_Count] > 0 then
            Warnings & {"Was_Reopened=false but Reopen_Count>0"}
        else Warnings,
        
        // Check First_Reopen_Date consistency
        Warnings = if [Was_Reopened] = true and [First_Reopen_Date] = null then
            Warnings & {"Reopened ticket missing First_Reopen_Date"}
        else if [Was_Reopened] = false and [First_Reopen_Date] <> null then
            Warnings & {"Non-reopened ticket has First_Reopen_Date"}
        else Warnings,
        
        // Check Days_To_First_Reopen logic
        Warnings = if [Days_To_First_Reopen] <> null and [Days_To_First_Reopen] < 0 then
            Warnings & {"Negative Days_To_First_Reopen"}
        else Warnings
    in
        if List.Count(Warnings) > 0 then Text.Combine(Warnings, "; ") else null)
```

---

## 6. Issue Resolution KPI Integration

### 6.1 Primary KPIs Using Was_Reopened

#### First-Pass Resolution Rate
```dax
First_Pass_Resolution_Rate = 
DIVIDE(
    COUNTROWS(
        FILTER(
            Fact_Ticket_Summary,
            [IsResolved] = TRUE && [Was_Reopened] = FALSE
        )
    ),
    COUNTROWS(FILTER(Fact_Ticket_Summary, [IsResolved] = TRUE)),
    0
) * 100
```

#### Reopened Rate
```dax
ReopenedRate = 
DIVIDE(
    COUNTROWS(FILTER(Fact_Ticket_Summary, [Was_Reopened] = TRUE)),
    COUNTROWS(Fact_Ticket_Summary),
    0
) * 100
```

#### Quality-Adjusted SLO
```dax
Quality_Adjusted_SLO_Rate = 
DIVIDE(
    COUNTROWS(
        FILTER(
            Fact_Ticket_Summary,
            [IsResolved] = TRUE && 
            [ResponseTimeWithinSLO] = TRUE && 
            [Was_Reopened] = FALSE
        )
    ),
    COUNTROWS(
        FILTER(
            Fact_Ticket_Summary,
            [IsResolved] = TRUE && [Was_Reopened] = FALSE
        )
    ),
    0
) * 100
```

### 6.2 Advanced Analytics

#### Resolution Efficiency Ratio
```dax
Resolution_Efficiency_Ratio = 
VAR TotalEffort = 
    SUMX(
        Fact_Ticket_Summary,
        [TotalResponseTimeHours] + 
        IF([Was_Reopened] = TRUE, [TotalResponseTimeHours] * [Reopen_Count] * 0.3, 0)
    )
VAR IdealEffort = 
    SUMX(Fact_Ticket_Summary, [TotalResponseTimeHours])
RETURN
    DIVIDE(IdealEffort, TotalEffort, 1)
```

#### Capability Resolution Maturity
```dax
Capability_Resolution_Maturity = 
VAR FirstPassRate = [First_Pass_Resolution_Rate]
VAR AvgReopenTime = [Avg_Days_To_First_Reopen]
VAR ReopenFrequency = [Avg_Reopening_Frequency]

VAR MaturityScore = 
    (FirstPassRate * 0.5) + 
    (IF(AvgReopenTime > 7, 25, 0) * 0.3) + 
    (IF(ReopenFrequency < 1.5, 25, 0) * 0.2)

RETURN
    SWITCH(
        TRUE(),
        MaturityScore >= 90, "Mature",
        MaturityScore >= 70, "Developing", 
        MaturityScore >= 50, "Basic",
        "Needs Improvement"
    )
```

---

## 7. Dashboard Integration

### 7.1 Executive Dashboard Components

**Key Visualizations Enabled:**
1. **Resolution Quality Scorecard**
   - First-Pass Resolution Rate
   - Quality-Adjusted SLO Achievement
   - Resolution Stability Score

2. **Trends and Patterns**
   - Monthly reopening rate trends
   - Capability-level resolution maturity
   - Process improvement velocity

3. **Operational Insights**
   - Root cause analysis of reopenings
   - Predictive quality indicators
   - Team performance comparisons

### 7.2 Drill-Down Capabilities

**From Executive to Operational:**
- Capability → Service → Individual tickets
- Time period → Monthly → Daily → Hourly patterns
- Issue type → Priority → Assignee performance

**From Metric to Action:**
- High reopening rate → Root cause analysis
- Quality decline → Process review recommendations
- Anomaly detection → Immediate investigation prompts

---

## 8. Implementation Benefits

### 8.1 Operational Benefits
- **Accurate Performance Measurement**: Eliminates false positives in resolution metrics
- **Proactive Quality Management**: Identifies systemic issues before they become widespread
- **Resource Optimization**: Quantifies hidden cost of rework and optimizes allocation

### 8.2 Strategic Benefits
- **Process Excellence**: Drives continuous improvement culture
- **Customer Experience**: Reduces frustration from recurring issues
- **Accountability**: Provides clear metrics for team and individual performance

### 8.3 Governance Benefits
- **Data-Driven Decisions**: Supports evidence-based process improvements
- **Predictive Analytics**: Enables proactive quality management
- **Compliance**: Provides audit trail for resolution quality

---

## 9. Future Enhancements

### 9.1 Planned Improvements
1. **Machine Learning Integration**: Predict tickets likely to be reopened
2. **Real-Time Alerting**: Immediate notification of quality issues
3. **Advanced Root Cause Analysis**: Natural language processing of reopen reasons
4. **Customer Feedback Integration**: Correlate reopenings with satisfaction scores

### 9.2 Scalability Considerations
- **Performance Optimization**: Implement aggregation tables for large datasets
- **Cross-Platform Integration**: Extend to other ticketing systems (ServiceNow, etc.)
- **Organizational Rollout**: Scale validation framework to additional capabilities

---

## 10. Conclusion

The implementation of reopened ticket detection and tracking provides a robust foundation for measuring true issue resolution quality. By distinguishing between tickets resolved correctly on the first attempt versus those requiring additional work, the organization can implement accurate, actionable KPIs that drive continuous improvement and exceptional customer service.

The comprehensive validation framework ensures data accuracy and reliability, while the integration with existing SLO metrics provides a complete picture of service delivery performance. This implementation positions the organization to achieve sustainable improvements in issue resolution quality while maintaining operational efficiency and customer satisfaction.

---

## Technical Summary

| Component | Implementation Method | Key Fields Added |
|-----------|----------------------|------------------|
| **Fact_Ticket_Status_Change** | Power Query M | `ReopenEvent`, `ReopeningType`, `ReopenContext` |
| **Fact_Ticket_Summary** | Power Query M with nested table aggregation | `Was_Reopened`, `Reopen_Count`, `First_Reopen_Date`, `Days_To_First_Reopen` |
| **DAX Measures** | Calculated measures | `First_Pass_Resolution_Rate`, `ReopenedRate`, `Quality_Adjusted_SLO_Rate` |
| **Validation** | Both Power Query and DAX | Test cases, statistical monitoring, data quality checks |
| **Dashboard Integration** | Native Power BI visualizations | Executive scorecards, trend analysis, drill-down capabilities |