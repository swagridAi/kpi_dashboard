# Throughput Measure Dependencies Validation Guide

## Critical Dependencies Overview

### 1. Fact Table Structure
**Required Table**: `Fact_Ticket_Summary` (Note: Referenced as Fact_JiraTickets in original request)

**Required Columns**:
- `IsResolved` (Boolean) - Indicates ticket completion status
- `ResolvedDate` (DateTime) - Date when ticket was completed
- `CreatedDate` (DateTime) - Date when ticket was created
- `issue_type` (Text) - For grouping and filtering
- `key` (Text) - Unique ticket identifier

### 2. Date Dimension
**Required Table**: `Dim_Date`

**Required Columns**:
- `Date` (Date) - Primary date field for relationships
- `Year` (Integer) - For yearly aggregation
- `Month` (Integer) - For monthly aggregation
- `MonthStart` (Date) - Month boundary calculations
- `MonthEnd` (Date) - Month boundary calculations

### 3. Table Relationships
**Primary Relationships**:
- `Fact_Ticket_Summary[CreatedDate] ↔ Dim_Date[Date]` (Active)
- `Fact_Ticket_Summary[ResolvedDate] ↔ Dim_Date[Date]` (Inactive)

---

## Dependency Validation Steps

### 1. Fact Table Validation

```dax
-- Check if Fact_Ticket_Summary table exists
Validate_Fact_Table_Exists = 
IF(
    ISBLANK(COUNTROWS(Fact_Ticket_Summary)),
    "❌ Fact_Ticket_Summary table not found",
    "✅ Fact_Ticket_Summary table exists"
)

-- Verify required columns exist
Validate_IsResolved_Column = 
VAR TestColumn = 
    EVALUATE 
    ROW("Test", MAX(Fact_Ticket_Summary[IsResolved]))
RETURN
IF(
    ISERROR(TestColumn),
    "❌ IsResolved column missing or wrong data type",
    "✅ IsResolved column exists"
)

-- Check data quality of IsResolved column
IsResolved_Data_Quality = 
VAR TotalRows = COUNTROWS(Fact_Ticket_Summary)
VAR RowsWithResolved = 
    COUNTROWS(
        FILTER(
            Fact_Ticket_Summary,
            NOT ISBLANK(Fact_Ticket_Summary[IsResolved])
        )
    )
VAR CompletionRate = DIVIDE(RowsWithResolved, TotalRows) * 100
RETURN
IF(
    CompletionRate >= 95,
    "✅ IsResolved data quality: " & FORMAT(CompletionRate, "0.0%"),
    "⚠️ IsResolved data quality: " & FORMAT(CompletionRate, "0.0%") & " (< 95%)"
)

-- Validate resolved tickets have resolution dates
Validate_ResolutionDate_Consistency = 
VAR ResolvedTickets = 
    COUNTROWS(
        FILTER(
            Fact_Ticket_Summary,
            Fact_Ticket_Summary[IsResolved] = TRUE
        )
    )
VAR ResolvedWithDates = 
    COUNTROWS(
        FILTER(
            Fact_Ticket_Summary,
            Fact_Ticket_Summary[IsResolved] = TRUE &&
            NOT ISBLANK(Fact_Ticket_Summary[ResolvedDate])
        )
    )
VAR ConsistencyRate = DIVIDE(ResolvedWithDates, ResolvedTickets) * 100
RETURN
"Resolution Date Consistency: " & FORMAT(ConsistencyRate, "0.0%")
```

### 2. Date Dimension Validation

```dax
-- Check if Dim_Date exists and has required coverage
Validate_Date_Table = 
VAR MinTicketDate = MIN(Fact_Ticket_Summary[CreatedDate])
VAR MaxTicketDate = MAX(Fact_Ticket_Summary[ResolvedDate])
VAR MinDimDate = MIN(Dim_Date[Date])
VAR MaxDimDate = MAX(Dim_Date[Date])
RETURN
IF(
    MinDimDate <= MinTicketDate && MaxDimDate >= MaxTicketDate,
    "✅ Date dimension covers all ticket dates",
    "❌ Date dimension coverage insufficient"
)

-- Validate date dimension completeness (no gaps)
Validate_Date_Continuity = 
VAR DateCount = COUNTROWS(Dim_Date)
VAR ExpectedDays = 
    DATEDIFF(MIN(Dim_Date[Date]), MAX(Dim_Date[Date]), DAY) + 1
RETURN
IF(
    DateCount = ExpectedDays,
    "✅ Date dimension has no gaps",
    "⚠️ Date dimension may have gaps: " & DateCount & " vs " & ExpectedDays
)

-- Check for required date dimension columns
Validate_Date_Columns = 
VAR HasYear = NOT ISERROR(MAX(Dim_Date[Year]))
VAR HasMonth = NOT ISERROR(MAX(Dim_Date[Month]))
VAR HasMonthStart = NOT ISERROR(MAX(Dim_Date[MonthStart]))
VAR HasMonthEnd = NOT ISERROR(MAX(Dim_Date[MonthEnd]))
RETURN
"Year: " & IF(HasYear, "✅", "❌") & 
" | Month: " & IF(HasMonth, "✅", "❌") &
" | MonthStart: " & IF(HasMonthStart, "✅", "❌") &
" | MonthEnd: " & IF(HasMonthEnd, "✅", "❌")
```

### 3. Relationship Validation

```dax
-- Test active relationship (CreatedDate)
Test_CreatedDate_Relationship = 
VAR TestResult = 
    CALCULATE(
        COUNTROWS(Fact_Ticket_Summary),
        Dim_Date[Year] = 2024
    )
RETURN
IF(
    TestResult > 0,
    "✅ CreatedDate relationship working",
    "❌ CreatedDate relationship broken"
)

-- Test inactive relationship (ResolvedDate)
Test_ResolvedDate_Relationship = 
VAR TestResult = 
    CALCULATE(
        COUNTROWS(Fact_Ticket_Summary),
        USERELATIONSHIP(Fact_Ticket_Summary[ResolvedDate], Dim_Date[Date]),
        Dim_Date[Year] = 2024
    )
RETURN
IF(
    TestResult > 0,
    "✅ ResolvedDate relationship working",
    "❌ ResolvedDate relationship broken"
)

-- Validate relationship cardinality
Validate_Relationship_Cardinality = 
VAR TicketsPerDate = 
    AVERAGEX(
        VALUES(Dim_Date[Date]),
        CALCULATE(COUNTROWS(Fact_Ticket_Summary))
    )
RETURN
"Average tickets per date: " & FORMAT(TicketsPerDate, "0.00")
```

### 4. Data Type Validation

```sql
-- SQL script to validate data types in source database
SELECT 
    COLUMN_NAME,
    DATA_TYPE,
    IS_NULLABLE,
    CASE 
        WHEN COLUMN_NAME = 'IsResolved' AND DATA_TYPE = 'bit' THEN '✅ Correct'
        WHEN COLUMN_NAME = 'ResolvedDate' AND DATA_TYPE IN ('datetime', 'datetime2') THEN '✅ Correct'
        WHEN COLUMN_NAME = 'CreatedDate' AND DATA_TYPE IN ('datetime', 'datetime2') THEN '✅ Correct'
        ELSE '❌ Check Type'
    END AS Validation_Status
FROM INFORMATION_SCHEMA.COLUMNS
WHERE TABLE_NAME = 'Fact_Ticket_Summary'
  AND COLUMN_NAME IN ('IsResolved', 'ResolvedDate', 'CreatedDate', 'issue_type', 'key');
```

### 5. Measure Dependencies Test

```dax
-- Test basic throughput calculation
Test_Basic_Throughput = 
VAR BasicCount = 
    COUNTROWS(
        FILTER(
            Fact_Ticket_Summary,
            Fact_Ticket_Summary[IsResolved] = TRUE
        )
    )
VAR MeasureResult = [Throughput]
RETURN
IF(
    BasicCount = MeasureResult,
    "✅ Throughput measure working correctly",
    "❌ Throughput measure error: Expected " & BasicCount & ", Got " & MeasureResult
)

-- Test time intelligence dependencies
Test_Time_Intelligence = 
VAR HasPreviousMonth = NOT ISBLANK([Previous_Month_Throughput])
VAR HasMoMChange = NOT ISBLANK([MoM_Throughput_Change_Percent])
RETURN
"Previous Month: " & IF(HasPreviousMonth, "✅", "❌") &
" | MoM Change: " & IF(HasMoMChange, "✅", "❌")
```

---

## Manual Validation Checklist

### Power BI Model View Checks

1. **Table Existence**
   - [ ] Fact_Ticket_Summary visible in Model view
   - [ ] Dim_Date visible in Model view

2. **Column Verification**
   - [ ] IsResolved column shows as Boolean (True/False icon)
   - [ ] ResolvedDate shows as DateTime (calendar icon)
   - [ ] CreatedDate shows as DateTime (calendar icon)

3. **Relationship Verification**
   - [ ] Line connects CreatedDate to Dim_Date[Date] (solid line = active)
   - [ ] Line connects ResolvedDate to Dim_Date[Date] (dashed line = inactive)
   - [ ] Relationships show One-to-Many cardinality (1:*)
   - [ ] Cross-filter direction set correctly

### Data Quality Checks

```dax
-- Create a validation dashboard with these measures
Validation_Summary = 
"Fact Table: " & [Validate_Fact_Table_Exists] &
UNICHAR(10) & "IsResolved Column: " & [Validate_IsResolved_Column] &
UNICHAR(10) & "Date Coverage: " & [Validate_Date_Table] &
UNICHAR(10) & "CreatedDate Rel: " & [Test_CreatedDate_Relationship] &
UNICHAR(10) & "ResolvedDate Rel: " & [Test_ResolvedDate_Relationship] &
UNICHAR(10) & "Throughput Test: " & [Test_Basic_Throughput]
```

### Performance Validation

```dax
-- Test measure performance with large date ranges
Performance_Test = 
VAR StartTime = NOW()
VAR TestResult = [Throughput]
VAR EndTime = NOW()
VAR ExecutionTime = DATEDIFF(StartTime, EndTime, SECOND)
RETURN
"Throughput: " & TestResult & " | Time: " & ExecutionTime & "s"
```

## Common Issues and Solutions

### Issue 1: Missing IsResolved Column
**Symptoms**: Error referencing IsResolved
**Solution**: 
- Check source table schema
- Verify column name (might be 'is_resolved', 'resolved', or 'status' = 'Done')
- Update measure to use correct column name

### Issue 2: Relationship Not Working
**Symptoms**: Throughput returns same value regardless of date selection
**Solution**:
- Delete and recreate relationships
- Verify date column data types match
- Check for blank/null values in date columns

### Issue 3: Inconsistent Monthly Totals
**Symptoms**: Monthly aggregations don't match detail records
**Solution**:
- Ensure using USERELATIONSHIP for ResolvedDate
- Validate month boundary calculations
- Check for timezone issues in date columns

### Issue 4: Performance Problems
**Symptoms**: Slow measure calculation
**Solution**:
- Create calculated columns for IsResolved if coming from complex logic
- Consider aggregation tables for historical data
- Optimize relationships and remove unnecessary columns

## Dependency Verification Report Template

Create a report page with:
1. **Card visuals** showing each validation measure
2. **Table visual** showing sample data with all required columns
3. **Line chart** testing the Throughput measure over time
4. **Matrix visual** testing monthly aggregation

This comprehensive validation ensures the Throughput measure will work reliably across all scenarios in your SLO dashboard.
