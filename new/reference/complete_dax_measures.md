Complete DAX Measures for SLO Dashboard (Simplified)
Overview
This document provides the complete set of DAX measures for the simplified SLO Dashboard system, focusing exclusively on 6 core KPIs that provide essential service performance insights:

Lead Time - Time from creation to work start
Cycle Time - Time from work start to completion
Response Time - End-to-end resolution time
Throughput - Volume of completed tickets
Service Quality - SLO achievement percentage
Issue Resolution Time - Average resolution time
The simplified approach removes complex analytics, individual performance tracking, priority adjustments, and complex reopening detection, focusing on actionable metrics that drive service improvement.

Core KPI Measures
1. Service Quality KPI (SLO Achievement Rate)
dax
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
2. SLA Target Calculation (Simplified Hierarchy)
dax
SLA_Target_Days = 
VAR IssueType = SELECTEDVALUE(Fact_Ticket_Summary[issue_type])
VAR SLA_Target = 
    COALESCE(
        -- Capability-level SLA
        CALCULATE(
            MAX(Dim_Capability[ResponseTimeTargetDays]),
            Config_Issue_Type_Capability_Mapping[IssueType] = IssueType
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
3. SLA Achievement Calculation (Simplified)
dax
Met_SLA = 
VAR ActualResolutionDays = Fact_Ticket_Summary[ResolutionTimeDays]
VAR IsResolved = Fact_Ticket_Summary[IsResolved]
VAR IssueType = Fact_Ticket_Summary[issue_type]

-- Get SLA target using simplified fallback approach
VAR SLA_Target = 
    COALESCE(
        -- Capability-level target
        CALCULATE(
            RELATED(Dim_Capability[ResponseTimeTargetDays]),
            USERELATIONSHIP(
                Fact_Ticket_Summary[issue_type], 
                Config_Issue_Type_Capability_Mapping[IssueType]
            )
        ),
        
        -- Default SLA table
        CALCULATE(
            MAX(Default_SLA_Table[SLA_Days]),
            USERELATIONSHIP(Fact_Ticket_Summary[issue_type], Default_SLA_Table[TicketType]),
            Default_SLA_Table[IsActive] = TRUE
        ),
        
        -- Ultimate fallback
        5
    )

RETURN 
    SWITCH(
        TRUE(),
        NOT IsResolved, BLANK(),                    -- Cannot determine yet
        ActualResolutionDays <= SLA_Target, TRUE,  -- Met SLA
        ActualResolutionDays > SLA_Target, FALSE,  -- Missed SLA
        BLANK()                                     -- Default case
    )
4. Lead Time (KPI #1)
dax
Lead_Time_Days = 
CALCULATE(
    AVERAGE(Fact_Ticket_Status_Change[DurationBusinessHours]),
    Fact_Ticket_Status_Change[IsLeadTimeStart] = TRUE
) / 24
5. Cycle Time (KPI #2)
dax
Cycle_Time_Days = 
VAR CycleTimeTickets = 
    SUMMARIZE(
        FILTER(Fact_Ticket_Status_Change, [IsCycleTimeStart] = TRUE),
        [TicketKey],
        "CycleTime", SUM(Fact_Ticket_Status_Change[DurationBusinessHours])
    )
RETURN AVERAGEX(CycleTimeTickets, [CycleTime]) / 24
6. Response Time (KPI #3)
dax
Avg_Response_Time_Days = 
CALCULATE(
    AVERAGE(Fact_Ticket_Summary[ResolutionTimeDays]),
    Fact_Ticket_Summary[IsResolved] = TRUE
)
7. Throughput (KPI #4)
dax
Throughput_KPI = 
CALCULATE(
    COUNTROWS(Fact_Ticket_Summary),
    Fact_Ticket_Summary[Is_Completed] = TRUE,
    USERELATIONSHIP(Fact_Ticket_Summary[ResolvedDate], Dim_Date[Date])
)
8. Issue Resolution Time (KPI #6)
dax
Issue_Resolution_Time = 
CALCULATE(
    AVERAGE(Fact_Ticket_Summary[ResolutionTimeDays]),
    Fact_Ticket_Summary[IsResolved] = TRUE
)
Supporting Volume Measures
9. Monthly Throughput
dax
Monthly_Throughput = 
CALCULATE(
    [Throughput_KPI],
    DATESMTD(Dim_Date[Date])
)
10. Average Daily Throughput
dax
Avg_Daily_Throughput = 
VAR DaysInPeriod = DISTINCTCOUNT(Dim_Date[Date])
VAR TotalThroughput = [Throughput_KPI]
RETURN DIVIDE(TotalThroughput, DaysInPeriod, 0)
11. Total Tickets Created
dax
Total_Tickets_Created = 
COUNTROWS(Fact_Ticket_Summary)
12. Total Open Tickets
dax
Open_Tickets = 
COUNTROWS(
    FILTER(
        Fact_Ticket_Summary,
        [IsResolved] = FALSE
    )
)
13. Tickets At Risk
dax
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
Time Intelligence Measures
14. Month-over-Month SLO Change
dax
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
15. Six-Month Average SLO
dax
Six_Month_Avg_SLO = 
CALCULATE(
    [SLO_Achievement_Rate],
    DATESINPERIOD(Dim_Date[Date], MAX(Dim_Date[Date]), -6, MONTH)
)
16. Year-over-Year SLO Change
dax
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
17. Quarter-to-Date SLO
dax
QTD_SLO_Achievement = 
CALCULATE(
    [SLO_Achievement_Rate],
    DATESQTD(Dim_Date[Date])
)
18. Year-to-Date SLO
dax
YTD_SLO_Achievement = 
CALCULATE(
    [SLO_Achievement_Rate],
    DATESYTD(Dim_Date[Date])
)
19. Rolling 3-Month SLO
dax
Rolling_3_Month_SLO = 
CALCULATE(
    [SLO_Achievement_Rate],
    DATESINPERIOD(Dim_Date[Date], MAX(Dim_Date[Date]), -3, MONTH)
)
Capability-Level Measures (Simplified)
20. Capability SLO Performance
dax
Capability_SLO_Performance = 
CALCULATE(
    [SLO_Achievement_Rate],
    USERELATIONSHIP(
        Fact_Ticket_Summary[issue_type], 
        Config_Issue_Type_Capability_Mapping[IssueType]
    )
)
21. Total Tickets by Capability
dax
Total_Tickets_By_Capability = 
CALCULATE(
    COUNTROWS(Fact_Ticket_Summary),
    USERELATIONSHIP(
        Fact_Ticket_Summary[issue_type], 
        Config_Issue_Type_Capability_Mapping[IssueType]
    )
)
22. Capability Throughput
dax
Capability_Throughput = 
CALCULATE(
    [Throughput_KPI],
    USERELATIONSHIP(
        Fact_Ticket_Summary[issue_type], 
        Config_Issue_Type_Capability_Mapping[IssueType]
    )
)
Trend Analysis Measures
23. SLO Trend Direction
dax
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
Validation and Quality Measures
24. Data Quality Check
dax
Data_Quality_Check = 
VAR TotalRecords = COUNTROWS(Fact_Ticket_Summary)
VAR CompleteRecords = 
    COUNTROWS(
        FILTER(
            Fact_Ticket_Summary,
            NOT ISBLANK([created]) &&
            NOT ISBLANK([issue_type]) &&
            NOT ISBLANK([status])
        )
    )
VAR Completeness = DIVIDE(CompleteRecords, TotalRecords, 0) * 100

VAR InvalidRecords = 
    COUNTROWS(
        FILTER(
            Fact_Ticket_Summary,
            [ResolutionTimeDays] < 0 ||
            ([IsResolved] = TRUE && ISBLANK([resolution_date]))
        )
    )
VAR Accuracy = (1 - DIVIDE(InvalidRecords, TotalRecords, 0)) * 100

RETURN 
    "Completeness: " & FORMAT(Completeness, "0.0%") & 
    " | Accuracy: " & FORMAT(Accuracy, "0.0%")
25. SLO Calculation Test
dax
SLO_Calculation_Test = 
VAR TestTickets = 
    FILTER(
        Fact_Ticket_Summary,
        [key] IN {"TEST-001", "TEST-002", "TEST-003"}
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
26. Data Freshness Check
dax
Data_Freshness_Check = 
VAR LastRefresh = MAX(Fact_Ticket_Summary[updated])
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
Utility and Helper Measures
27. Current Month Tickets
dax
Current_Month_Tickets = 
CALCULATE(
    COUNTROWS(Fact_Ticket_Summary),
    DATESMTD(Dim_Date[Date])
)
28. Business Days in Period
dax
Business_Days_In_Period = 
CALCULATE(
    COUNTROWS(Dim_Date),
    Dim_Date[IsBusinessDay] = TRUE
)
29. SLO Variance Days
dax
SLO_Variance_Days = 
AVERAGEX(
    FILTER(Fact_Ticket_Summary, [IsResolved] = TRUE),
    [ResolutionTimeDays] - [SLA_Target_Days]
)
30. Team Performance Summary
dax
Team_Performance_Summary = 
VAR SLORate = [SLO_Achievement_Rate]
VAR AvgDays = [Avg_Response_Time_Days]
VAR Throughput = [Throughput_KPI]

RETURN 
    "SLO: " & FORMAT(SLORate, "0.0%") & 
    " | Avg Days: " & FORMAT(AvgDays, "0.1") &
    " | Tickets: " & FORMAT(Throughput, "0")
Simple Calculated Columns
31. Days_In_Current_Status (Calculated Column)
dax
Days_In_Current_Status = 
DATEDIFF(
    Fact_Ticket_Summary[updated], 
    TODAY(), 
    DAY
)
32. SLA_Status (Calculated Column)
dax
SLA_Status = 
SWITCH(
    TRUE(),
    [resolution_date] = BLANK() AND [ResolutionTimeDays] > [SLA_Target_Days], "Open - SLA Breached",
    [resolution_date] = BLANK() AND [ResolutionTimeDays] > [SLA_Target_Days] * 0.8, "Open - At Risk",
    [resolution_date] = BLANK(), "Open - Within SLA", 
    [Met_SLA] = TRUE, "Resolved - SLA Met",
    [Met_SLA] = FALSE, "Resolved - SLA Missed",
    "Unknown"
)
33. Resolution_Time_Business_Days (Calculated Column)
dax
Resolution_Time_Business_Days = 
VAR CreatedDate = Fact_Ticket_Summary[created]
VAR ResolvedDate = 
    IF(
        ISBLANK(Fact_Ticket_Summary[resolution_date]),
        TODAY(),
        Fact_Ticket_Summary[resolution_date]
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

KPI Mapping Reference
Core KPI	Primary Measure	Supporting Measures
1. Lead Time	Lead_Time_Days	Time intelligence variants
2. Cycle Time	Cycle_Time_Days	Time intelligence variants
3. Response Time	Avg_Response_Time_Days	SLO_Variance_Days
4. Throughput	Throughput_KPI	Monthly_Throughput, Avg_Daily_Throughput
5. Service Quality	SLO_Achievement_Rate	Met_SLA, SLA_Target_Days
6. Issue Resolution	Issue_Resolution_Time	Resolution_Time_Business_Days
Implementation Notes
Simplified Architecture:
2-tier SLA hierarchy: Capability → Default fallback only
No priority adjustments: All tickets use standard SLA targets
No individual tracking: Focus on team-level performance
Basic time intelligence: Essential trend analysis without advanced forecasting
Key Relationships:
Fact_Ticket_Summary ↔ Dim_Date (created/resolved)
Fact_Ticket_Summary ↔ Config_Issue_Type_Capability_Mapping
Config_Issue_Type_Capability_Mapping ↔ Dim_Capability
Fact_Ticket_Status_Change ↔ Fact_Ticket_Summary
Validation Requirements:
Ensure all measures handle null values appropriately
Test SLA hierarchy with various ticket configurations
Verify business day calculations align with organizational calendar
Confirm time intelligence measures work across different date ranges
This simplified DAX implementation provides comprehensive coverage of the 6 core KPIs while maintaining the simplicity needed for effective service performance monitoring.

