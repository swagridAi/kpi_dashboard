# Key Performance Indicators Overview

The SLO Dashboard tracks six key performance indicators that provide a comprehensive view of service delivery across speed, volume, quality, and sustainability dimensions. These KPIs evolved from basic SLO tracking to create a balanced measurement system that prevents teams from optimizing one metric at the expense of others.

Why six KPIs instead of just one? Each metric reveals different aspects of service performance:
- **Time-based metrics** (Lead, Cycle, Response Time) show how quickly work flows through your process
- **Volume metrics** (Throughput) reveal capacity and delivery consistency  
- **Quality metrics** (Service Quality, Issue Resolution) ensure speed doesn't sacrifice effectiveness

Understanding these KPIs helps teams measure what matters most to their customers while providing the data needed for continuous improvement.

## SLO KPIs

These three time-based metrics form the foundation of service level measurement:

### Lead Time

**What it measures:** How quickly your team begins working on new requests

**Business Definition:** The time from when a ticket is created (or enters your backlog) until active work begins. This measures your team's responsiveness to new customer needs.

**Typical targets:** 1-2 days for urgent requests, 3-5 days for standard work, 5-7 days for low-priority items.

**Why it matters:**
- Shows how quickly you respond to customer requests
- Indicates capacity availability and prioritization effectiveness
- Helps set realistic expectations for customers
- Reveals bottlenecks in intake and triage processes

**Real-world example:** A Data Quality team receives a data validation request on Monday. If they begin analysis on Wednesday, the lead time is 2 days. Teams use this to identify when their intake process creates delays.

### Cycle Time

**What it measures:** How efficiently you complete work once started

**Business Definition:** The time from when work begins (ticket moves to "In Progress") until completion. This measures your team's execution efficiency within their control.

**Typical targets:** 2-4 days for standard requests, 1-2 days for simple tasks, 5-10 days for complex work.

**Why it matters:**
- Indicates process efficiency and team expertise
- Helps identify workflow bottlenecks and improvement opportunities
- Enables accurate delivery forecasting once work begins
- Supports resource planning and capacity management

**Real-world example:** Once the Data Quality team starts their validation work on Wednesday, they complete it by Friday—a 2-day cycle time. This metric helps them optimize their workflow and predict completion dates.

### Response Time

**What it measures:** Total customer experience from request to delivery

**Business Definition:** The complete end-to-end time from ticket creation until final resolution. This represents the full customer journey and forms the basis for SLA commitments.

**Typical targets:** 3-5 days for bugs, 5-7 days for standard tasks, 10-15 days for complex projects.

**Why it matters:**
- Directly impacts customer satisfaction and trust
- Reflects overall service delivery performance  
- Forms the basis for service level agreements with customers
- Enables comparison with industry benchmarks and competitor performance

**Real-world example:** From Monday's data validation request to Friday's delivery, the complete response time is 5 days. This is what customers experience and what SLA commitments are based on.

**Relationship to Lead + Cycle:** Response Time = Lead Time + Cycle Time + any delays. Understanding this breakdown helps teams identify whether delays occur before work starts (lead time) or during execution (cycle time).

## Volume and Quality KPIs

These metrics balance speed measurements with delivery effectiveness:

### Throughput

**What it measures:** The volume of work completed over time

**Business Definition:** The number of tickets completed per day, week, or month, providing insight into team capacity and delivery volume.

**Business value:**
- **Capacity Planning:** Understand realistic delivery rates for future planning
- **Resource Allocation:** Identify teams needing additional support or redistribution
- **Bottleneck Identification:** Spot workflow stages where work accumulates
- **Performance Trending:** Track whether delivery capacity is improving or declining

**Usage scenarios:** Teams use throughput to predict delivery timelines, justify resource requests, and identify when process changes impact delivery volume. Executives use it for capacity planning and team performance comparison.

### Service Quality

**What it measures:** How effectively services meet customer needs within committed timeframes

**Business Definition:** The percentage of tickets resolved within their SLA timeframe, measuring both speed and commitment adherence. Calculated as: (Tickets meeting SLA ÷ Total resolved tickets) × 100.

**Performance targets:** Organizations typically target 90-95% service quality, with 95%+ considered excellent performance.

**Why it matters:**
- Ensures speed improvements don't compromise delivery effectiveness
- Builds customer trust through consistent delivery performance
- Prevents the hidden costs of missed commitments and rework
- Provides objective measure of service reliability

**Real-world example:** If a team resolves 95 tickets within their SLA targets out of 100 completed tickets, their service quality score is 95%. This metric catches teams that might be completing work quickly but missing their promised delivery dates.

### Issue Resolution

**What it measures:** The completeness and durability of problem-solving

**Business Definition:** Two key aspects: (1) Average time to final resolution, and (2) Resolution stability—the percentage of tickets that remain closed after initial resolution.

**Performance targets:** Less than 15% of tickets should require reopening, with average resolution times meeting established SLA targets.

**Why it matters:**
- **Quality Indicator:** Tickets that stay resolved indicate effective problem-solving and knowledge application
- **Hidden Cost Prevention:** Reopened tickets consume 2-3× more resources than initial resolution due to context rebuilding and customer frustration
- **Process Maturity:** Low reopening rates suggest mature processes, effective knowledge sharing, and proper root cause analysis
- **Customer Experience:** Stable resolutions prevent repeated customer contacts and relationship strain

**Real-world example:** A team resolves 100 tickets in a month. If 12 tickets get reopened for additional work, their resolution stability is 88%. The total effort includes not just initial resolution time, but also the additional time spent on reopened tickets.

**Cost implications:** Organizations often underestimate the true cost of poor resolution quality. Reopened tickets require additional investigation, re-establishing context, customer re-communication, and often involve multiple team members—making them significantly more expensive than getting it right the first time.

## How KPIs Work Together

These six KPIs create a balanced measurement system that encourages sustainable, high-quality performance:

**The Complete Picture:**
- **Speed Metrics** (Lead, Cycle, Response Time) → Show how fast work flows
- **Volume Metrics** (Throughput) → Show how much work gets done
- **Quality Metrics** (Service Quality, Issue Resolution) → Show how well work is done

**Preventing Gaming:**
Teams can't optimize one metric by sacrificing others because:
- Fast response times with poor service quality indicate rushing or unrealistic SLA targets
- High throughput with increasing reopening rates suggests corner-cutting
- Excellent cycle times alongside declining lead times reveals capacity constraints
- Good service quality with low throughput may indicate over-engineering or inefficient processes

**Balanced Scorecard Benefits:**
- **Sustainable Performance:** Teams must maintain quality while improving speed
- **Holistic Optimization:** Improvements consider the entire service delivery process
- **Early Warning System:** Degradation in one metric often predicts issues in others
- **Stakeholder Alignment:** Different audiences can focus on relevant metrics while understanding the complete picture

**Real-world Balance Example:**
A team improving their cycle time from 5 to 3 days (good) but experiencing increased reopening rates from 10% to 25% (concerning) indicates they're rushing work. The balanced view prevents celebrating the cycle time improvement without addressing the quality issue.

## Interpreting Your KPI Dashboard

**Healthy Performance Indicators:**
- **SLO Achievement:** 90-95%+ service quality rates sustained over time
- **Stable Throughput:** Consistent delivery volume without quality degradation
- **Low Reopening:** Less than 15% of tickets requiring additional work
- **Balanced Metrics:** Improvements across KPIs rather than excellence in just one area
- **Predictable Patterns:** Steady performance with explainable variations

**Warning Signs to Watch:**
- **Quality Trade-offs:** Improving response times with rising reopening rates
- **Volume vs. Quality:** High throughput accompanied by declining service quality
- **Capacity Issues:** Excellent cycle times but deteriorating lead times (backlog building)
- **Inconsistent Performance:** Wide variation between team members or issue types
- **Customer Satisfaction Disconnect:** Good metrics but increasing customer complaints

**Common Performance Patterns:**
- **New Teams:** Start with longer lead/cycle times that improve as processes mature; may initially struggle with consistency
- **Mature Teams:** Display stable metrics with gradual improvements; rarely experience dramatic swings
- **Overloaded Teams:** Show increasing lead times first, followed by declining throughput, then quality degradation
- **Process Changes:** Create temporary disruption across all metrics before stabilizing at new performance levels
- **Seasonal Patterns:** Predictable variations based on business cycles, holiday periods, or budget cycles

**Using Multiple KPIs for Diagnosis:**
- **High lead time + normal cycle time** = Capacity or prioritization issue
- **Normal lead time + high cycle time** = Process efficiency or complexity issue  
- **Good time metrics + poor quality** = Skills gap or pressure to deliver quickly
- **Declining throughput + stable times** = Increasing work complexity or external dependencies

## Next Steps

Ready to use these KPIs effectively?

- **View Your Dashboard:** Learn how to interpret your team's specific metrics → [Team Operations Guide](../dashboard-usage/team-operations.md)
- **Set Up Alerts:** Get notified when KPIs indicate issues → [Alerts and Subscriptions](../dashboard-usage/alerts-and-subscriptions.md)
- **Configure Targets:** Customize KPI targets for your team → [Basic Configuration](../getting-started/basic-configuration.md)
- **Executive View:** Understand organization-wide KPI performance → [Executive Views](../dashboard-usage/executive-views.md)

Understanding these KPIs is the foundation for data-driven service improvement. They provide the common language needed to discuss performance, identify opportunities, and demonstrate value across your organization.