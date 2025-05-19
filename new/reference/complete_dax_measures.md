# Complete DAX Measures for SLO Dashboard

## Core SLO Measures

### 1. SLO Achievement Rate
```dax
SLO_Achievement_Rate = 
VAR ResolvedTickets = 
    FILTER(
        Fact_Ticket_Summary,
        [IsResolved] = TRUE && NOT ISBLANK([Met_SLA])
    )
VAR TicketsMetSLA = 
    FILTER(ResolvedTickets, [Met_SLA] = TRUE)
RETURN
    DIVIDE(COUNTROWS(TicketsMetSLA), COUNTROWS(ResolvedTickets), 0) * 100
```

### 2. Met_SLA Calculation
```dax
Met_SLA = 
VAR ActualResolutionDays = Fact_Ticket_Summary[ResolutionTimeDays]
VAR IsResolved = Fact_Ticket_Summary[IsResolved]
VAR IssueType = Fact_Ticket_Summary[IssueType]

-- Get SLA target using hierarchical fallback
VAR SLA_Target = 
    COALESCE(
        -- Priority 1: Service-specific override
        RELATED(Dim_Service[ServiceResponseTimeTarget]),
        
        -- Priority 2: Capability-level target
        CALCULATE(
            RELATED(Dim_Capability[ResponseTimeTargetDays]),
            USERELATIONSHIP(
                Fact_Ticket_Summary[IssueType], 
                Config_Issue_Type_Mapping[IssueType]
            )
        ),
        
        -- Priority 3: Default SLA table
        CALCULATE(
            MAX(Default_SLA_Table[SLA_Days]),
            USERELATIONSHIP(Fact_Ticket_Summary[IssueType], Default_SLA_Table[TicketType]),
            Default_SLA_Table[IsActive] = TRUE
        ),
        
        -- Priority 4: Ultimate fallback
        5
    )

-- Apply priority adjustments if configured
VAR PriorityMultiplier = 
    SWITCH(
        Fact_Ticket_Summary[Priority],
        "P1", 0.5,
        "P2", 0.75, 
        "P3", 1.0,
        "P4", 1.5,
        1.0
    )

VAR AdjustedSLA = SLA_Target * PriorityMultiplier

RETURN 
    SWITCH(
        TRUE(),
        NOT IsResolved, BLANK(),                    -- Cannot determine yet
        ActualResolutionDays <= AdjustedSLA, TRUE, -- Met SLA
        ActualResolutionDays > AdjustedSLA, FALSE, -- Missed SLA
        BLANK()                                     -- Default case
    )
```

### 3. SLA Target Days
```dax
SLA_Target_Days = 
VAR IssueType = SELECTEDVALUE(Fact_Ticket_Summary[IssueType])
VAR SLA_Target = 
    COALESCE(
        -- Service-specific SLA override
        RELATED(Dim_Service[ServiceResponseTimeTarget]),
        
        -- Capability-level SLA
        CALCULATE(
            MAX(Dim_Capability[ResponseTimeTargetDays]),
            Config_Issue_Type_Mapping[IssueType] = IssueType
        ),
        
        -- Default SLA fallback
        CALCULATE(
            MAX(Default_SLA_Table[SLA_Days]),
            Default_SLA_Table[TicketType] = IssueType,
            Default_SLA_Table[IsActive] = TRUE
        ),
        
        -- Ultimate fallback
        5
    )
RETURN SLA_Target
```

## Time-Based Measures

### 4. Lead Time Average
```dax
Lead_Time_Days = 
CALCULATE(
    AVERAGE(Fact_Status_Change[DurationBusinessHours]),
    Fact_Status_Change[IsLeadTimeStart] = TRUE
) / 24
```

### 5. Cycle Time Average
```dax
Cycle_Time_Days = 
VAR CycleTimeTickets = 
    SUMMARIZE(
        FILTER(Fact_Status_Change, [IsCycleTimeStart] = TRUE),
        [TicketKey],
        "CycleTime", SUM(Fact_Status_Change[DurationBusinessHours])
    )
RETURN AVERAGEX(CycleTimeTickets, [CycleTime]) / 24
```

### 6. Response Time Average
```dax
Avg_Response_Time_Days = 
CALCULATE(
    AVERAGE(Fact_Ticket_Summary[ResolutionTimeDays]),
    Fact_Ticket_Summary[IsResolved] = TRUE
)
```

### 7. Response Time Median
```dax
Median_Response_Time_Days = 
CALCULATE(
    MEDIAN(Fact_Ticket_Summary[ResolutionTimeDays]),
    Fact_Ticket_Summary[IsResolved] = TRUE
)
```

### 8. Response Time 90th Percentile
```dax
P90_Response_Time_Days = 
CALCULATE(
    PERCENTILE.INC(Fact_Ticket_Summary[ResolutionTimeDays], 0.9),
    Fact_Ticket_Summary[IsResolved] = TRUE
)
```

## Volume and Throughput Measures

### 9. Throughput
```dax
Throughput_KPI = 
CALCULATE(
    COUNTROWS(Fact_Ticket_Summary),
    Fact_Ticket_Summary[Is_Completed] = TRUE,
    USERELATIONSHIP(Fact_Ticket_Summary[ResolvedDate], Dim_Date[Date])
)
```

### 10. Monthly Throughput
```dax
Monthly_Throughput = 
CALCULATE(
    [Throughput_KPI],
    DATESMTD(Dim_Date[Date])
)
```

### 11. Average Daily Throughput
```dax
Avg_Daily_Throughput = 
VAR DaysInPeriod = DISTINCTCOUNT(Dim_Date[Date])
VAR TotalThroughput = [Throughput_KPI]
RETURN DIVIDE(TotalThroughput, DaysInPeriod, 0)
```

### 12. Total Tickets Created
```dax
Total_Tickets_Created = 
COUNTROWS(Fact_Ticket_Summary)
```

### 13. Total Open Tickets
```dax
Open_Tickets = 
COUNTROWS(
    FILTER(
        Fact_Ticket_Summary,
        [IsResolved] = FALSE
    )
)
```

### 14. Tickets At Risk
```dax
Tickets_At_Risk = 
VAR RiskThreshold = 0.8  -- 80% of SLA target
RETURN
COUNTROWS(
    FILTER(
        Fact_Ticket_Summary,
        [IsResolved] = FALSE &&
        [ResolutionTimeDays] >= [SLA_Target_Days] * RiskThreshold
    )
)
```

## Quality Measures

### 15. Service Quality KPI
```dax
Service_Quality_KPI = 
VAR TotalResolved = 
    COUNTROWS(
        FILTER(Fact_Ticket_Summary, [IsResolved] = TRUE)
    )
VAR QualityResolutions = 
    COUNTROWS(
        FILTER(
            Fact_Ticket_Summary, 
            [IsResolved] = TRUE && [Met_SLA] = TRUE
        )
    )
RETURN DIVIDE(QualityResolutions, TotalResolved, 0) * 100
```

### 16. First Pass Resolution Rate
```dax
First_Pass_Resolution_Rate = 
DIVIDE(
    COUNTROWS(
        FILTER(
            Fact_Ticket_Summary, 
            [IsResolved] = TRUE && [Was_Reopened] = FALSE
        )
    ),
    COUNTROWS(
        FILTER(Fact_Ticket_Summary, [IsResolved] = TRUE)
    ),
    0
) * 100
```

### 17. Reopening Rate
```dax
Reopening_Rate = 
DIVIDE(
    COUNTROWS(
        FILTER(Fact_Ticket_Summary, [Was_Reopened] = TRUE)
    ),
    COUNTROWS(
        FILTER(Fact_Ticket_Summary, [IsResolved] = TRUE)
    ),
    0
) * 100
```

### 18. Average Reopen Count
```dax
Avg_Reopen_Count = 
CALCULATE(
    AVERAGE(Fact_Ticket_Summary[Reopen_Count]),
    Fact_Ticket_Summary[Was_Reopened] = TRUE
)
```

## Time Intelligence Measures

### 19. Month-over-Month SLO Change
```dax
MoM_SLO_Change = 
VAR CurrentMonth = [SLO_Achievement_Rate]
VAR PreviousMonth = 
    CALCULATE(
        [SLO_Achievement_Rate],
        DATEADD(Dim_Date[Date], -1, MONTH)
    )
RETURN 
    IF(
        NOT ISBLANK(PreviousMonth) && PreviousMonth <> 0,
        CurrentMonth - PreviousMonth,
        BLANK()
    )
```

### 20. Six-Month Average SLO
```dax
Six_Month_Avg_SLO = 
CALCULATE(
    [SLO_Achievement_Rate],
    DATESINPERIOD(Dim_Date[Date], MAX(Dim_Date[Date]), -6, MONTH)
)
```

### 21. Year-over-Year SLO Change
```dax
YoY_SLO_Change = 
VAR CurrentPeriod = [SLO_Achievement_Rate]
VAR PreviousYear = 
    CALCULATE(
        [SLO_Achievement_Rate],
        DATEADD(Dim_Date[Date], -12, MONTH)
    )
RETURN 
    IF(
        NOT ISBLANK(PreviousYear) && PreviousYear <> 0,
        DIVIDE(CurrentPeriod - PreviousYear, PreviousYear) * 100,
        BLANK()
    )
```

### 22. Quarter-to-Date SLO
```dax
QTD_SLO_Achievement = 
CALCULATE(
    [SLO_Achievement_Rate],
    DATESQTD(Dim_Date[Date])
)
```

### 23. Year-to-Date SLO
```dax
YTD_SLO_Achievement = 
CALCULATE(
    [SLO_Achievement_Rate],
    DATESYTD(Dim_Date[Date])
)
```

### 24. Rolling 3-Month SLO
```dax
Rolling_3_Month_SLO = 
CALCULATE(
    [SLO_Achievement_Rate],
    DATESINPERIOD(Dim_Date[Date], MAX(Dim_Date[Date]), -3, MONTH)
)
```

## Capability-Level Measures

### 25. Capability SLO Performance
```dax
Capability_SLO_Performance = 
CALCULATE(
    [SLO_Achievement_Rate],
    USERELATIONSHIP(
        Fact_Ticket_Summary[IssueType], 
        Config_Issue_Type_Mapping[IssueType]
    )
)
```

### 26. Total Tickets by Capability
```dax
Total_Tickets_By_Capability = 
CALCULATE(
    COUNTROWS(Fact_Ticket_Summary),
    USERELATIONSHIP(
        Fact_Ticket_Summary[IssueType], 
        Config_Issue_Type_Mapping[IssueType]
    )
)
```

### 27. Capability Throughput
```dax
Capability_Throughput = 
CALCULATE(
    [Throughput_KPI],
    USERELATIONSHIP(
        Fact_Ticket_Summary[IssueType], 
        Config_Issue_Type_Mapping[IssueType]
    )
)
```

### 28. Capability Performance Score
```dax
Capability_Performance_Score = 
VAR SLOComponent = [SLO_Achievement_Rate] * 0.4
VAR QualityComponent = [Service_Quality_KPI] * 0.3
VAR ThroughputComponent = MIN([Throughput_KPI] / 20, 1) * 100 * 0.2
VAR StabilityComponent = (100 - [Reopening_Rate]) * 0.1

VAR RawScore = SLOComponent + QualityComponent + ThroughputComponent + StabilityComponent
VAR NormalizedScore = MIN(MAX(RawScore, 0), 100)

RETURN NormalizedScore
```

### 29. Capability Maturity Level
```dax
Capability_Maturity_Level = 
VAR Score = [Capability_Performance_Score]
RETURN 
    SWITCH(
        TRUE(),
        Score >= 90, "🌟 Excellent",
        Score >= 80, "✅ Good",
        Score >= 70, "⚠️ Developing",
        Score >= 60, "🔄 Fair",
        "❌ Needs Improvement"
    )
```

## Priority and Assignee Measures

### 30. High Priority SLO Rate
```dax
High_Priority_SLO_Rate = 
CALCULATE(
    [SLO_Achievement_Rate],
    Fact_Ticket_Summary[Priority] IN {"P1", "High", "Critical"}
)
```

### 31. Priority Distribution
```dax
Priority_High_Percentage = 
VAR HighPriorityTickets = 
    COUNTROWS(
        FILTER(Fact_Ticket_Summary, [Priority] IN {"P1", "High", "Critical"})
    )
VAR TotalTickets = COUNTROWS(Fact_Ticket_Summary)
RETURN DIVIDE(HighPriorityTickets, TotalTickets, 0) * 100
```

### 32. Assignee Performance
```dax
Assignee_SLO_Performance = 
CALCULATE(
    [SLO_Achievement_Rate],
    ALLEXCEPT(Fact_Ticket_Summary, Fact_Ticket_Summary[AssigneeDisplayName])
)
```

### 33. Team Workload Balance
```dax
Team_Workload_Balance = 
VAR MaxTickets = 
    CALCULATE(
        MAX(
            CALCULATETABLE(
                SUMMARIZE(
                    Fact_Ticket_Summary,
                    [AssigneeDisplayName],
                    "TicketCount", COUNTROWS(Fact_Ticket_Summary)
                )
            )
        ),
        ALL(Fact_Ticket_Summary[AssigneeDisplayName])
    )
VAR MinTickets = 
    CALCULATE(
        MIN(
            CALCULATETABLE(
                SUMMARIZE(
                    Fact_Ticket_Summary,
                    [AssigneeDisplayName],
                    "TicketCount", COUNTROWS(Fact_Ticket_Summary)
                )
            )
        ),
        ALL(Fact_Ticket_Summary[AssigneeDisplayName])
    )
VAR AvgTickets = DIVIDE(MaxTickets + MinTickets, 2)
VAR BalanceRatio = DIVIDE(MinTickets, MaxTickets, 0) * 100

RETURN BalanceRatio
```

## Trend and Forecasting Measures

### 34. SLO Trend Direction
```dax
SLO_Trend_Direction = 
VAR Current = [SLO_Achievement_Rate]
VAR Previous = 
    CALCULATE(
        [SLO_Achievement_Rate],
        DATEADD(Dim_Date[Date], -1, MONTH)
    )
VAR Direction = 
    SWITCH(
        TRUE(),
        ISBLANK(Previous), "📊 New",
        Current > Previous + 2, "📈 Improving",
        Current < Previous - 2, "📉 Declining",
        "➡️ Stable"
    )
RETURN Direction
```

### 35. SLO Forecast Next Month
```dax
SLO_Forecast_Next_Month = 
VAR HistoricalData = 
    CALCULATETABLE(
        SUMMARIZE(
            Fact_Ticket_Summary,
            Dim_Date[MonthStart],
            "SLO_Performance", [SLO_Achievement_Rate]
        ),
        DATESINPERIOD(Dim_Date[Date], MAX(Dim_Date[Date]), -6, MONTH)
    )

VAR Count = COUNTROWS(HistoricalData)
VAR SimpleForecast = 
    IF(
        Count >= 3,
        AVERAGEX(
            TOPN(3, HistoricalData, [MonthStart], DESC),
            [SLO_Performance]
        ),
        [Six_Month_Avg_SLO]
    )

RETURN SimpleForecast
```

### 36. Seasonal Adjustment Factor
```dax
Seasonal_Adjustment_Factor = 
VAR CurrentMonth = MONTH(MAX(Dim_Date[Date]))
VAR SeasonalMultiplier = 
    SWITCH(
        CurrentMonth,
        12, 1.2,  -- Holiday season
        1, 1.1,   -- January complexity
        7, 0.9,   -- Summer efficiency
        8, 0.9,   -- Summer efficiency
        1.0       -- Standard months
    )
RETURN SeasonalMultiplier
```

## Validation and Quality Measures

### 37. Data Quality Check
```dax
Data_Quality_Check = 
VAR TotalRecords = COUNTROWS(Fact_Ticket_Summary)
VAR CompleteRecords = 
    COUNTROWS(
        FILTER(
            Fact_Ticket_Summary,
            NOT ISBLANK([Created]) &&
            NOT ISBLANK([IssueType]) &&
            NOT ISBLANK([Status])
        )
    )
VAR Completeness = DIVIDE(CompleteRecords, TotalRecords, 0) * 100

VAR InvalidRecords = 
    COUNTROWS(
        FILTER(
            Fact_Ticket_Summary,
            [ResolutionTimeDays] < 0 ||
            ([IsResolved] = TRUE && ISBLANK([ResolutionDate]))
        )
    )
VAR Accuracy = (1 - DIVIDE(InvalidRecords, TotalRecords, 0)) * 100

RETURN 
    "Completeness: " & FORMAT(Completeness, "0.0%") & 
    " | Accuracy: " & FORMAT(Accuracy, "0.0%")
```

### 38. SLO Calculation Test
```dax
SLO_Calculation_Test = 
VAR TestTickets = 
    FILTER(
        Fact_Ticket_Summary,
        [TicketKey] IN {"TEST-001", "TEST-002", "TEST-003"}
    )
VAR ValidationCount = 
    SUMX(
        TestTickets,
        IF([Met_SLA] <> BLANK(), 1, 0)
    )
RETURN 
    IF(
        ValidationCount > 0,
        "✅ SLO calculations working",
        "❌ SLO calculation issues detected"
    )
```

### 39. Relationship Validation
```dax
Relationship_Validation = 
VAR OrphanedTickets = 
    COUNTROWS(
        FILTER(
            Fact_Ticket_Summary,
            ISBLANK(RELATED(Dim_Date[Date]))
        )
    )
VAR TotalTickets = COUNTROWS(Fact_Ticket_Summary)
VAR OrphanRate = DIVIDE(OrphanedTickets, TotalTickets, 0) * 100

RETURN 
    IF(
        OrphanRate = 0, 
        "✅ All relationships intact",
        "⚠️ " & FORMAT(OrphanRate, "0.0%") & " orphaned records"
    )
```

### 40. Data Freshness Check
```dax
Data_Freshness_Check = 
VAR LastRefresh = MAX(Fact_Ticket_Summary[Updated])
VAR CurrentTime = NOW()
VAR HoursSinceRefresh = DATEDIFF(LastRefresh, CurrentTime, HOUR)
VAR Status = 
    SWITCH(
        TRUE(),
        HoursSinceRefresh <= 24, "🟢 Current",
        HoursSinceRefresh <= 48, "🟡 Slightly Stale",
        HoursSinceRefresh <= 72, "🟠 Stale",
        "🔴 Very Stale"
    )

RETURN 
    "Last Update: " & FORMAT(LastRefresh, "MMM DD, YYYY HH:MM") & 
    " (" & HoursSinceRefresh & "h ago) - " & Status
```

## Utility and Helper Measures

### 41. Current Month Tickets
```dax
Current_Month_Tickets = 
CALCULATE(
    COUNTROWS(Fact_Ticket_Summary),
    DATESMTD(Dim_Date[Date])
)
```

### 42. Business Days in Period
```dax
Business_Days_In_Period = 
CALCULATE(
    COUNTROWS(Dim_Date),
    Dim_Date[IsBusinessDay] = TRUE
)
```

### 43. SLO Variance Days
```dax
SLO_Variance_Days = 
AVERAGEX(
    FILTER(Fact_Ticket_Summary, [IsResolved] = TRUE),
    [ResolutionTimeDays] - [SLA_Target_Days]
)
```

### 44. Performance Index
```dax
Performance_Index = 
VAR SLOIndex = [SLO_Achievement_Rate] / 95 * 100  -- Normalize to 95% target
VAR QualityIndex = [Service_Quality_KPI] / 90 * 100  -- Normalize to 90% target
VAR ThroughputIndex = [Throughput_KPI] / 25 * 100  -- Normalize to 25 tickets target

VAR CompositeIndex = (SLOIndex * 0.5) + (QualityIndex * 0.3) + (ThroughputIndex * 0.2)
RETURN MIN(CompositeIndex, 150)  -- Cap at 150%
```

### 45. Team Performance Summary
```dax
Team_Performance_Summary = 
VAR SLORate = [SLO_Achievement_Rate]
VAR AvgDays = [Avg_Response_Time_Days]
VAR Throughput = [Throughput_KPI]
VAR Quality = [Service_Quality_KPI]

RETURN 
    "SLO: " & FORMAT(SLORate, "0.0%") & 
    " | Avg Days: " & FORMAT(AvgDays, "0.1") &
    " | Tickets: " & FORMAT(Throughput, "0") &
    " | Quality: " & FORMAT(Quality, "0.0%")
```

## Calculated Columns

### 46. Was_Reopened (Calculated Column)
```dax
Was_Reopened = 
VAR TicketKey = Fact_Ticket_Summary[TicketKey]
VAR ReopenEvents = 
    CALCULATE(
        COUNTROWS(Fact_Status_Change),
        Fact_Status_Change[TicketKey] = TicketKey,
        Fact_Status_Change[ReopenEvent] = TRUE
    )
RETURN ReopenEvents > 0
```

### 47. Days_In_Current_Status (Calculated Column)
```dax
Days_In_Current_Status = 
DATEDIFF(
    Fact_Ticket_Summary[Updated], 
    TODAY(), 
    DAY
)
```

### 48. SLA_Status (Calculated Column)
```dax
SLA_Status = 
SWITCH(
    TRUE(),
    [ResolutionDate] = BLANK() AND [ResolutionTimeDays] > [SLA_Target_Days], "Open - SLA Breached",
    [ResolutionDate] = BLANK() AND [ResolutionTimeDays] > [SLA_Target_Days] * 0.8, "Open - At Risk",
    [ResolutionDate] = BLANK(), "Open - Within SLA", 
    [Met_SLA] = TRUE, "Resolved - SLA Met",
    [Met_SLA] = FALSE, "Resolved - SLA Missed",
    "Unknown"
)
```

### 49. Complexity_Score (Calculated Column)
```dax
Complexity_Score = 
VAR StatusChanges = 
    CALCULATE(
        COUNTROWS(Fact_Status_Change),
        Fact_Status_Change[TicketKey] = Fact_Ticket_Summary[TicketKey]
    )
VAR Score = 
    SWITCH(
        TRUE(),
        StatusChanges <= 3, 1,  -- Simple
        StatusChanges <= 6, 2,  -- Medium
        StatusChanges <= 10, 3, -- Complex
        4                       -- Very Complex
    )
RETURN Score
```

### 50. Resolution_Time_Business_Days (Calculated Column)
```dax
Resolution_Time_Business_Days = 
VAR CreatedDate = Fact_Ticket_Summary[Created]
VAR ResolvedDate = 
    IF(
        ISBLANK(Fact_Ticket_Summary[ResolutionDate]),
        TODAY(),
        Fact_Ticket_Summary[ResolutionDate]
    )

-- Calculate business days between dates
VAR BusinessDays = 
    CALCULATE(
        COUNTROWS(Dim_Date),
        Dim_Date[Date] >= CreatedDate,
        Dim_Date[Date] <= ResolvedDate,
        Dim_Date[IsBusinessDay] = TRUE
    )

RETURN BusinessDays
```