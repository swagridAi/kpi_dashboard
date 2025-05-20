// ==========================================================
// SLO DASHBOARD – COMPLETE, CLEANED-UP DAX SCRIPT (34 objects)
// ==========================================================
// • All “dax” prefixes removed
// • All comments use // or /* … */
// • Format strings fixed (mm = minutes, MM = month)
// • Lead/Cycle time now divide after capturing hours
// • Business-day filter wrapped in KEEPFILTERS
// Copy-paste each definition into Power BI (or DAX Studio) one by one.
// ==========================================================



// ----------------------------------------------------------
// 1  SERVICE QUALITY KPI  (SLO ACHIEVEMENT RATE)
// ----------------------------------------------------------
SLO_Achievement_Rate =
VAR ResolvedTickets =
    FILTER (
        Fact_Ticket_Summary,
        Fact_Ticket_Summary[IsResolved] = TRUE ()
            && NOT ISBLANK ( [Met_SLA] )
    )
VAR TicketsMetSLA =
    FILTER ( ResolvedTickets, [Met_SLA] = TRUE )
RETURN
    DIVIDE ( COUNTROWS ( TicketsMetSLA ),
             COUNTROWS ( ResolvedTickets ),
             0
    ) * 100



// ----------------------------------------------------------
// 2  SLA TARGET (DAYS)  – 2-tier hierarchy
// ----------------------------------------------------------
SLA_Target_Days =
VAR IssueType = SELECTEDVALUE ( Fact_Ticket_Summary[issue_type] )
VAR SLA_Target =
    COALESCE (
        // Capability-level SLA
        CALCULATE (
            MAX ( Dim_Capability[ResponseTimeTargetDays] ),
            Config_Issue_Type_Capability_Mapping[issue_type] = IssueType
        ),

        // Default SLA fallback
        CALCULATE (
            MAX ( Default_SLA_Table[SLA_Days] ),
            Default_SLA_Table[TicketType] = IssueType,
            Default_SLA_Table[IsActive] = TRUE ()
        ),

        // Ultimate fallback
        5
    )
RETURN
    SLA_Target



// ----------------------------------------------------------
// 3  MET / MISS SLA  (BOOLEAN)
// ----------------------------------------------------------
Met_SLA =
VAR ActualResolutionDays = Fact_Ticket_Summary[ResolutionTimeDays]
VAR IsResolved          = Fact_Ticket_Summary[IsResolved]
VAR IssueType           = Fact_Ticket_Summary[issue_type]

VAR SLA_Target =
    COALESCE (
        // Capability-level target
        CALCULATE (
            RELATED ( Dim_Capability[ResponseTimeTargetDays] ),
            USERELATIONSHIP (
                Fact_Ticket_Summary[issue_type],
                Config_Issue_Type_Capability_Mapping[issue_type]
            )
        ),

        // Default SLA table
        CALCULATE (
            MAX ( Default_SLA_Table[SLA_Days] ),
            USERELATIONSHIP (
                Fact_Ticket_Summary[issue_type],
                Default_SLA_Table[TicketType]
            ),
            Default_SLA_Table[IsActive] = TRUE ()
        ),

        // Ultimate fallback
        5
    )

RETURN
    SWITCH (
        TRUE (),
        NOT IsResolved,                                  BLANK (),
        ActualResolutionDays <= SLA_Target,              TRUE (),
        ActualResolutionDays >  SLA_Target,              FALSE (),
        BLANK ()
    )



// ----------------------------------------------------------
// 4  LEAD TIME – DAYS
// ----------------------------------------------------------
Lead_Time_Days =
VAR Hours =
    CALCULATE (
        AVERAGE ( Fact_Ticket_Status_Change[DurationBusinessHours] ),
        Fact_Ticket_Status_Change[IsLeadTimeStart] = TRUE ()
    )
RETURN
    DIVIDE ( Hours, 24 )



// ----------------------------------------------------------
// 5  CYCLE TIME – DAYS
// ----------------------------------------------------------
Cycle_Time_Days =
VAR CycleTimeTickets =
    SUMMARIZE (
        FILTER (
            Fact_Ticket_Status_Change,
            Fact_Ticket_Status_Change[IsCycleTimeStart] = TRUE ()
        ),
        Fact_Ticket_Status_Change[key],
        "CycleTimeHours",
            SUM ( Fact_Ticket_Status_Change[DurationBusinessHours] )
    )
VAR AvgHours = AVERAGEX ( CycleTimeTickets, [CycleTimeHours] )
RETURN
    DIVIDE ( AvgHours, 24 )



// ----------------------------------------------------------
// 6  RESPONSE TIME – AVG DAYS
// ----------------------------------------------------------
Avg_Response_Time_Days =
CALCULATE (
    AVERAGE ( Fact_Ticket_Summary[ResolutionTimeDays] ),
    Fact_Ticket_Summary[IsResolved] = TRUE ()
)



// ----------------------------------------------------------
// 7  THROUGHPUT  (completed tickets)
// ----------------------------------------------------------
Throughput_KPI =
CALCULATE (
    COUNTROWS ( Fact_Ticket_Summary ),
    Fact_Ticket_Summary[Is_Completed] = TRUE (),
    USERELATIONSHIP ( Fact_Ticket_Summary[ResolvedDate], Dim_Date[Date] )
)



// ----------------------------------------------------------
// 8  ISSUE RESOLUTION TIME – AVG DAYS
// ----------------------------------------------------------
Issue_Resolution_Time =
CALCULATE (
    AVERAGE ( Fact_Ticket_Summary[ResolutionTimeDays] ),
    Fact_Ticket_Summary[IsResolved] = TRUE ()
)



// ----------------------------------------------------------
// 9  MONTHLY THROUGHPUT
// ----------------------------------------------------------
Monthly_Throughput =
CALCULATE ( [Throughput_KPI], DATESMTD ( Dim_Date[Date] ) )



// ----------------------------------------------------------
// 10  AVERAGE DAILY THROUGHPUT
// ----------------------------------------------------------
Avg_Daily_Throughput =
VAR DaysInPeriod   = DISTINCTCOUNT ( Dim_Date[Date] )
VAR TotalThroughput = [Throughput_KPI]
RETURN
    DIVIDE ( TotalThroughput, DaysInPeriod, 0 )



// ----------------------------------------------------------
// 11  TOTAL TICKETS CREATED
// ----------------------------------------------------------
Total_Tickets_Created = COUNTROWS ( Fact_Ticket_Summary )



// ----------------------------------------------------------
// 12  TOTAL OPEN TICKETS
// ----------------------------------------------------------
Open_Tickets =
COUNTROWS (
    FILTER ( Fact_Ticket_Summary, Fact_Ticket_Summary[IsResolved] = FALSE () )
)



// ----------------------------------------------------------
// 13  TICKETS AT RISK  (≥80 % of SLA)
// ----------------------------------------------------------
Tickets_At_Risk =
VAR RiskThreshold = 0.8
RETURN
    COUNTROWS (
        FILTER (
            Fact_Ticket_Summary,
            Fact_Ticket_Summary[IsResolved] = FALSE ()
                && Fact_Ticket_Summary[ResolutionTimeDays]
                     >= [SLA_Target_Days] * RiskThreshold
        )
    )



// ----------------------------------------------------------
// 14  MONTH-OVER-MONTH  SLO Δ (% pts)
// ----------------------------------------------------------
MoM_SLO_Change =
VAR CurrentMonth  = [SLO_Achievement_Rate]
VAR PreviousMonth =
    CALCULATE ( [SLO_Achievement_Rate],
                DATEADD ( Dim_Date[Date], -1, MONTH ) )
RETURN
    IF (
        NOT ISBLANK ( PreviousMonth ) && PreviousMonth <> 0,
        CurrentMonth - PreviousMonth
    )



// ----------------------------------------------------------
// 15  SIX-MONTH AVERAGE SLO
// ----------------------------------------------------------
Six_Month_Avg_SLO =
CALCULATE (
    [SLO_Achievement_Rate],
    DATESINPERIOD ( Dim_Date[Date], MAX ( Dim_Date[Date] ), -6, MONTH )
)



// ----------------------------------------------------------
// 16  YEAR-OVER-YEAR  SLO Δ  (% change)
// ----------------------------------------------------------
YoY_SLO_Change =
VAR CurrentPeriod = [SLO_Achievement_Rate]
VAR PreviousYear  =
    CALCULATE ( [SLO_Achievement_Rate],
                DATEADD ( Dim_Date[Date], -12, MONTH ) )
RETURN
    IF (
        NOT ISBLANK ( PreviousYear ) && PreviousYear <> 0,
        DIVIDE ( CurrentPeriod - PreviousYear, PreviousYear ) * 100
    )



// ----------------------------------------------------------
// 17  QUARTER-TO-DATE  SLO
// ----------------------------------------------------------
QTD_SLO_Achievement =
CALCULATE ( [SLO_Achievement_Rate], DATESQTD ( Dim_Date[Date] ) )



// ----------------------------------------------------------
// 18  YEAR-TO-DATE  SLO
// ----------------------------------------------------------
YTD_SLO_Achievement =
CALCULATE ( [SLO_Achievement_Rate], DATESYTD ( Dim_Date[Date] ) )



// ----------------------------------------------------------
// 19  ROLLING 3-MONTH  SLO
// ----------------------------------------------------------
Rolling_3_Month_SLO =
CALCULATE (
    [SLO_Achievement_Rate],
    DATESINPERIOD ( Dim_Date[Date], MAX ( Dim_Date[Date] ), -3, MONTH )
)



// ----------------------------------------------------------
// 20  CAPABILITY-LEVEL  SLO PERFORMANCE
// ----------------------------------------------------------
Capability_SLO_Performance =
CALCULATE (
    [SLO_Achievement_Rate],
    USERELATIONSHIP (
        Fact_Ticket_Summary[issue_type],
        Config_Issue_Type_Capability_Mapping[issue_type]
    )
)



// ----------------------------------------------------------
// 21  TOTAL TICKETS  BY CAPABILITY
// ----------------------------------------------------------
Total_Tickets_By_Capability =
CALCULATE (
    COUNTROWS ( Fact_Ticket_Summary ),
    USERELATIONSHIP (
        Fact_Ticket_Summary[issue_type],
        Config_Issue_Type_Capability_Mapping[issue_type]
    )
)



// ----------------------------------------------------------
// 22  CAPABILITY-LEVEL  THROUGHPUT
// ----------------------------------------------------------
Capability_Throughput =
CALCULATE (
    [Throughput_KPI],
    USERELATIONSHIP (
        Fact_Ticket_Summary[issue_type],
        Config_Issue_Type_Capability_Mapping[issue_type]
    )
)



// ----------------------------------------------------------
// 23  SLO TREND DIRECTION  (emoji label)
// ----------------------------------------------------------
SLO_Trend_Direction =
VAR Current  = [SLO_Achievement_Rate]
VAR Previous =
    CALCULATE ( [SLO_Achievement_Rate],
                DATEADD ( Dim_Date[Date], -1, MONTH ) )
VAR Direction =
    SWITCH (
        TRUE (),
        ISBLANK ( Previous ),         "📊 New",
        Current > Previous + 2,       "📈 Improving",
        Current < Previous - 2,       "📉 Declining",
        "➡️ Stable"
    )
RETURN
    Direction



// ----------------------------------------------------------
// 24  DATA QUALITY CHECK  – completeness & accuracy
// ----------------------------------------------------------
Data_Quality_Check =
VAR TotalRecords =
    COUNTROWS ( Fact_Ticket_Summary )

VAR CompleteRecords =
    COUNTROWS (
        FILTER (
            Fact_Ticket_Summary,
            NOT ISBLANK ( Fact_Ticket_Summary[created] )
                && NOT ISBLANK ( Fact_Ticket_Summary[issue_type] )
                && NOT ISBLANK ( Fact_Ticket_Summary[status] )
        )
    )
VAR Completeness =
    DIVIDE ( CompleteRecords, TotalRecords, 0 ) * 100

VAR InvalidRecords =
    COUNTROWS (
        FILTER (
            Fact_Ticket_Summary,
            Fact_Ticket_Summary[ResolutionTimeDays] < 0
                || ( Fact_Ticket_Summary[IsResolved] = TRUE ()
                     && ISBLANK ( Fact_Ticket_Summary[resolution_date] ) )
        )
    )
VAR Accuracy =
    ( 1 - DIVIDE ( InvalidRecords, TotalRecords, 0 ) ) * 100

RETURN
    "Completeness: "
        & FORMAT ( Completeness, "0.0%" )
        & " | Accuracy: "
        & FORMAT ( Accuracy, "0.0%" )



// ----------------------------------------------------------
// 25  SLO CALCULATION TEST  (quick regression)
// ----------------------------------------------------------
SLO_Calculation_Test =
VAR TestTickets =
    FILTER (
        Fact_Ticket_Summary,
        Fact_Ticket_Summary[key]
            IN { "TEST-001", "TEST-002", "TEST-003" }
    )
VAR ValidationCount =
    SUMX ( TestTickets, IF ( [Met_SLA] <> BLANK (), 1, 0 ) )
RETURN
    IF (
        ValidationCount > 0,
        "✅ SLO calculations working",
        "❌ SLO calculation issues detected"
    )



// ----------------------------------------------------------
// 26  DATA FRESHNESS CHECK  (last refresh status)
// ----------------------------------------------------------
Data_Freshness_Check =
VAR LastRefresh      = MAX ( Fact_Ticket_Summary[updated] )
VAR CurrentTime      = NOW ()
VAR HoursSinceRefresh = DATEDIFF ( LastRefresh, CurrentTime, HOUR )

VAR FreshnessStatus =
    SWITCH (
        TRUE (),
        HoursSinceRefresh <= 24, "🟢 Current",
        HoursSinceRefresh <= 48, "🟡 Slightly Stale",
        HoursSinceRefresh <= 72, "🟠 Stale",
        "🔴 Very Stale"
    )
RETURN
    "Last Update: "
        & FORMAT ( LastRefresh, "MMM dd, yyyy HH:mm" )
        & " (" & HoursSinceRefresh & "h ago) – "
        & FreshnessStatus



// ----------------------------------------------------------
// 27  CURRENT MONTH TICKETS
// ----------------------------------------------------------
Current_Month_Tickets =
CALCULATE (
    COUNTROWS ( Fact_Ticket_Summary ),
    DATESMTD ( Dim_Date[Date] )
)



// ----------------------------------------------------------
// 28  BUSINESS DAYS IN PERIOD
// ----------------------------------------------------------
Business_Days_In_Period =
CALCULATE (
    COUNTROWS ( Dim_Date ),
    KEEPFILTERS ( Dim_Date[IsBusinessDay] = TRUE () )
)



// ----------------------------------------------------------
// 29  SLO VARIANCE – AVG (resolution – target)
// ----------------------------------------------------------
SLO_Variance_Days =
AVERAGEX (
    FILTER ( Fact_Ticket_Summary, Fact_Ticket_Summary[IsResolved] = TRUE () ),
    Fact_Ticket_Summary[ResolutionTimeDays] - [SLA_Target_Days]
)



// ----------------------------------------------------------
// 30  TEAM PERFORMANCE SUMMARY  (single text cell)
// ----------------------------------------------------------
Team_Performance_Summary =
VAR SLORate    = [SLO_Achievement_Rate]
VAR AvgDays    = [Avg_Response_Time_Days]
VAR Throughput = [Throughput_KPI]
RETURN
    "SLO: "
        & FORMAT ( SLORate, "0.0%" )
        & " | Avg Days: "
        & FORMAT ( AvgDays, "0.1" )
        & " | Tickets: "
        & FORMAT ( Throughput, "0" )



// ==========================================================
// CALCULATED COLUMNS
// ==========================================================

// 31  CAPABILITY KEY  (direct look-up)
CapabilityKey =
RELATED ( Config_Issue_Type_Capability_Mapping[CapabilityKey] )



// 32  DAYS IN CURRENT STATUS
Days_In_Current_Status =
DATEDIFF ( Fact_Ticket_Summary[updated], TODAY (), DAY )



// 33  SLA STATUS (categorical)
// Open / At Risk / Breached / Met / Missed
SLA_Status =
SWITCH (
    TRUE (),
    Fact_Ticket_Summary[resolution_date] = BLANK ()
        && Fact_Ticket_Summary[ResolutionTimeDays]
             > [SLA_Target_Days],
            "Open – SLA Breached",

    Fact_Ticket_Summary[resolution_date] = BLANK ()
        && Fact_Ticket_Summary[ResolutionTimeDays]
             > [SLA_Target_Days] * 0.8,
            "Open – At Risk",

    Fact_Ticket_Summary[resolution_date] = BLANK (),
            "Open – Within SLA",

    [Met_SLA] = TRUE (),  "Resolved – SLA Met",
    [Met_SLA] = FALSE (), "Resolved – SLA Missed",
    "Unknown"
)



// 34  RESOLUTION TIME – BUSINESS DAYS
Resolution_Time_Business_Days =
VAR CreatedDate =
    Fact_Ticket_Summary[created]

VAR ResolvedDate =
    IF (
        ISBLANK ( Fact_Ticket_Summary[resolution_date] ),
        TODAY (),
        Fact_Ticket_Summary[resolution_date]
    )

VAR BusinessDays =
    CALCULATE (
        COUNTROWS ( Dim_Date ),
        Dim_Date[Date] >= CreatedDate,
        Dim_Date[Date] <= ResolvedDate,
        Dim_Date[IsBusinessDay] = TRUE ()
    )
RETURN
    BusinessDays

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
- 2-tier SLA hierarchy: Capability → Default fallback only
- No priority adjustments: All tickets use standard SLA targets
- No individual tracking: Focus on team-level performance
- Basic time intelligence: Essential trend analysis without advanced forecasting

Key Relationships:
- Fact_Ticket_Summary ↔ Dim_Date (created/resolved)
- Fact_Ticket_Summary ↔ Config_Issue_Type_Capability_Mapping
- Config_Issue_Type_Capability_Mapping ↔ Dim_Capability
- Fact_Ticket_Status_Change ↔ Fact_Ticket_Summary

Enhanced Capability Access:
- The CapabilityKey calculated column provides direct access to the capability
- This provides better filtering and slicing capabilities in reports
- Reduces the need for complex USERELATIONSHIP expressions in some contexts

Validation Requirements:
- Ensure all measures handle null values appropriately
- Test SLA hierarchy with various ticket configurations
- Verify business day calculations align with organizational calendar
- Confirm time intelligence measures work across different date ranges

This simplified DAX implementation provides comprehensive coverage of the 6 core KPIs while maintaining the simplicity needed for effective service performance monitoring.