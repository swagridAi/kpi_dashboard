# Service Quality KPI Implementation Summary

## What is the Service Quality KPI?

The Service Quality KPI measures **how well our team delivers on our service commitments** by tracking the percentage of completed tickets that were resolved within the agreed Service Level Agreement (SLA) timeframes.

**In simple terms:** *If we promise to fix a data quality issue within 3 days, and we actually fix it in 2 days, that counts toward our Service Quality score. If it takes 4 days, it doesn't.*

### Why This KPI Matters

- **Customer Trust**: Shows our reliability in meeting commitments
- **Team Performance**: Identifies areas where we excel or need improvement
- **Business Impact**: Demonstrates the value of our data services to the organization
- **Continuous Improvement**: Helps us focus on what needs fixing first

---

## How the Service Quality KPI Works

### The Basic Logic

1. **Track Ticket Lifecycle**: Monitor when tickets are created and when they're resolved
2. **Calculate Resolution Time**: Count the number of days from creation to resolution
3. **Compare to SLA Target**: Check if resolution time meets the SLA commitment
4. **Calculate Success Rate**: Count successful tickets as a percentage of all completed tickets

### Example Walkthrough

**Scenario**: Data Extract capability has an SLA of 3 days

- **Ticket A**: Created Monday, resolved Tuesday (1 day) ✅ **Meets SLA**
- **Ticket B**: Created Monday, resolved Thursday (3 days) ✅ **Meets SLA**  
- **Ticket C**: Created Monday, resolved Friday (4 days) ❌ **Misses SLA**

**Service Quality Score**: 2 out of 3 tickets met SLA = **67%**

---

## Technical Implementation Details

### New Data Fields Added

#### 1. ResolutionTimeDays
```
Purpose: Calculates how many days it took to resolve each ticket
Logic: End Date (Resolution) - Start Date (Creation) = Days
Result: 1.5 days, 3.2 days, etc.
```

#### 2. Met_SLA 
```
Purpose: Determines if a ticket met its SLA target
Logic: If ResolutionTimeDays ≤ SLA_Target_Days, then TRUE, else FALSE
Result: TRUE (met SLA) or FALSE (missed SLA)
```

#### 3. ServiceQuality Measure
```
Purpose: Calculates overall SLA adherence percentage
Logic: (Tickets meeting SLA ÷ Total completed tickets) × 100
Result: 87.5% (as a percentage)
```

### How SLA Targets Are Determined

1. **Issue Type Mapping**: Each ticket type (Bug, Story, Epic) is mapped to a capability
2. **Capability SLA**: Each capability has defined SLA targets (e.g., Data Quality = 3 days)
3. **Dynamic Lookup**: The system automatically applies the correct SLA based on ticket type

**Example Mapping**:
- Data Quality Bug → 2 days SLA
- Data Extract Request → 1 day SLA  
- Change Control Epic → 5 days SLA

---

## Data Model Dependencies

### Required Tables and Relationships

```
Fact_Ticket_Summary (Main ticket data)
    ↓ Connected to ↓
Config_Issue_Type_Capability_Mapping (Links ticket types to capabilities)
    ↓ Connected to ↓
Dim_Capability (Contains SLA targets for each capability)
```

### Key Data Requirements

- **Ticket Creation Date**: When the ticket was first created
- **Resolution Date**: When the ticket was marked as complete
- **Issue Type**: What kind of ticket it is (Bug, Story, Epic, etc.)
- **SLA Targets**: How many days each capability commits to for resolution

---

## Dashboard Display Recommendations

### 1. Executive Summary (KPI Card)

**What it shows**: One big number showing overall service quality
```
┌─────────────────┐
│  Service Quality │
│     92.5%       │
│  ↗ +2.1% vs LM  │
│  Target: 95%    │
└─────────────────┘
```

**Color coding**:
- 🟢 Green: 95%+ (Excellent)
- 🟡 Yellow: 85-94% (Good)  
- 🔴 Red: Below 85% (Needs Attention)

### 2. Detailed Breakdown (Bar Chart)

**What it shows**: Service quality by ticket type over the last 6 months

```
SLA Adherence by Ticket Type
   
100% |--|--|--|--|--|     Target Line (95%)
 80% |██|██|██|██|██|
 60% |██|██|██|██|██|
 40% |██|██|██|██|██|
 20% |██|██|██|██|██|
  0% +--+--+--+--+--+
     Bug Story Epic Incident Task
```

**Key insights to look for**:
- Which ticket types consistently miss SLA?
- Are there trending patterns over time?
- Which areas need the most attention?

### 3. Monthly Trend Analysis

**Purpose**: Track improvement or decline over time
- Line chart showing ServiceQuality % by month
- Helps identify seasonal patterns or process changes
- Shows impact of improvement initiatives

---

## How to Use This KPI

### For Capability Owners
- **Monitor your team's performance** against commitments
- **Identify problem ticket types** that frequently miss SLA
- **Set realistic SLA targets** based on historical performance
- **Celebrate improvements** and address declining performance

### For Executive Leadership  
- **Track overall service reliability** across all capabilities
- **Compare capability performance** to identify best practices
- **Make data-driven decisions** about resource allocation
- **Demonstrate value** to internal customers

### For Team Members
- **Understand expectations** for different types of work
- **Track personal contribution** to team SLA performance
- **Identify patterns** in tickets that take longer than expected
- **Focus efforts** on meeting commitments

---

## Success Criteria

### Target Performance Levels
- **Minimum Acceptable**: 85% of tickets meet SLA
- **Good Performance**: 90-94% of tickets meet SLA
- **Excellent Performance**: 95%+ of tickets meet SLA

### Improvement Actions if Below Target
1. **Analyze patterns**: Which ticket types or times miss SLA most?
2. **Process review**: Are our SLA targets realistic?
3. **Resource assessment**: Do we have adequate capacity?
4. **Training needs**: Are there skills gaps affecting delivery?

---

## Frequently Asked Questions

**Q: What if a ticket doesn't have an SLA target defined?**
A: It will be excluded from the ServiceQuality calculation but tracked separately.

**Q: Do weekends count toward resolution time?**
A: No, the system calculates business days only (Monday-Friday).

**Q: What if a ticket is reopened after being resolved?**
A: The original resolution time is maintained, and any reopening starts a new measurement period.

**Q: Can SLA targets be changed?**
A: Yes, but changes go through an approval process and take effect for new tickets only.

**Q: What about tickets that are still in progress?**
A: Only completed (resolved) tickets are included in the ServiceQuality calculation.

---

## Next Steps

1. **Validate the implementation** with sample data from each capability
2. **Train capability owners** on interpreting and using the KPI
3. **Establish review cadence** (monthly SLA performance reviews)
4. **Set improvement targets** for capabilities below 90%
5. **Create action plans** for systematic SLA improvement

---

*This Service Quality KPI provides a clear, measurable way to track and improve our service delivery performance, ensuring we consistently meet our commitments to internal customers while identifying opportunities for operational excellence.*