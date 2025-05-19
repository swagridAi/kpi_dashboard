# Quick Setup Guide

Get your team's SLO dashboard running in 15-30 minutes with default settings, then see immediate performance insights without complex configuration.

## Before You Begin

Gather this information:

- [ ] **Jira access** - You can view and search tickets for your team
- [ ] **Team identification** - Team name, capability owner, and main contact person  
- [ ] **Issue types** - List of Jira ticket types your team uses (Bug, Story, Task, Epic, etc.)
- [ ] **Service overview** - Brief description of what your team delivers

**What "immediate value" means:** Within 24 hours, you'll see your team's current SLO performance, trend indicators, and tickets at risk of missing deadlines - all with zero custom configuration required.

## 15-Minute Setup

### Step 1: Request Access (2 minutes)

**Submit your request:**
- **Email:** platform.support@company.com
- **Subject:** "SLO Dashboard Access Request - [Your Team Name]"
- **Required information:**
  - Team/capability name
  - Primary contact (name and email)
  - Brief service description (1-2 sentences)
  - List of Jira issue types you use

**Response timeline:** Access confirmation within 1 business day, including your personalized dashboard URL and Confluence configuration page link.

### Step 2: Complete Basic Configuration (8 minutes)

Access your team's configuration page using the Confluence link provided in your access confirmation email.

**2.1 Team Information (3 minutes)**
- **Capability name**: What your team is called (e.g., "Data Quality", "Data Extracts")
- **Team lead**: Primary contact for SLO-related decisions
- **Services offered**: Brief bullet list of what you deliver
- **Business domain**: Select from dropdown (Financial Products, Operations, etc.)

**2.2 Issue Type Mapping (3 minutes)**
- Select each Jira issue type your team uses
- Map each type to your capability using dropdown menus:
  - Bug → [Your Capability]
  - Story → [Your Capability]  
  - Task → [Your Capability]
- **Important**: This determines which tickets count toward your SLOs

**2.3 Default SLA Review (2 minutes)**
The system automatically applies these targets to get you started:
- **Bugs**: 3 business days
- **Stories**: 7 business days  
- **Tasks**: 5 business days
- **Epics**: 10 business days

*Note: You can customize these later. These defaults work for 80% of teams and provide immediate tracking.*

### Step 3: Verify Your Dashboard (5 minutes)

**Access and explore:**
1. **Open your dashboard** using the URL from your confirmation email
2. **Log in** with your company credentials (same as other internal tools)
3. **Navigate to your capability** using the menu or search
4. **Verify data appears** - you should see current tickets and basic metrics

**Quick validation checklist:**
- [ ] Your team's tickets appear in the dashboard
- [ ] SLO percentages show (may be 0% initially - that's normal)
- [ ] Current open tickets are listed
- [ ] Default SLA targets are displayed correctly

**Data refresh note:** New configurations take effect during the next nightly refresh (2:00 AM UTC). Same-day tickets will appear tomorrow with full SLO calculations.

**Initial dashboard elements:**
- **SLO Performance** - Current achievement percentage 
- **Trend Indicators** - Direction arrows (limited data initially)
- **Volume Metrics** - Count of tickets processed
- **At Risk Tickets** - Items approaching SLA deadlines
- **Recent Activity** - Latest ticket status changes

## What You'll See Over Time

**First 24 Hours:**
- Basic ticket counts and current status
- SLO percentages (may show 0% until first refresh completes)
- Open ticket list with time remaining until SLA deadline

**First Week:**
- Initial SLO performance percentages
- Basic volume trends
- Identification of tickets consistently missing SLA

**After 2-4 Weeks:**
- Meaningful trend lines and patterns
- Month-over-month performance comparisons
- Clear insights for process improvement

**Typical initial performance (for reference):**
- Most teams start between 65-85% SLO achievement
- Improvement typically seen within 4-6 weeks of active use
- Early alerts help prevent SLA breaches before they occur

## Immediate Next Steps

1. **Explore your dashboard** → [Understanding Your First Dashboard](first-dashboard.md)
2. **Customize settings** → [Basic Configuration Guide](basic-configuration.md)
3. **Set up alerts** → [Alerts and Subscriptions](../dashboard-usage/alerts-and-subscriptions.md)

## Getting Help

**For setup questions:** platform.support@company.com

**Common first-day issues:**

**Dashboard showing no data?**
- Configuration changes apply during nightly refresh (2:00 AM UTC)
- Verify issue type mappings in your Confluence page
- Ensure your Jira tickets are assigned to team members

**SLO percentages seem incorrect?**
- Check that issue types are mapped correctly
- Verify business day vs calendar day expectations
- Historical tickets will show corrected calculations after 24 hours

**Can't find your Confluence page?**
- Use the direct link from your access confirmation email
- Search Confluence for "[Your Team Name] SLO Configuration"
- Contact platform.support@company.com for the direct URL

**Need hands-on assistance?**
- **Monthly training sessions**: First Tuesday each month, 2:00 PM
- **One-on-one setup help**: Schedule via calendar link in confirmation email
- **Quick questions**: Email or Teams @SLO-Support for same-day response

---

**Quick Links:**
- **Dashboard Login**: [Use URL from your confirmation email]
- **Configuration Page**: [Direct link provided in access confirmation]
- **Support Email**: platform.support@company.com
- **Training Calendar**: [bookings.company.com/slo-training]

*Ready for the next step? Learn how to interpret your metrics in [Understanding Your First Dashboard](first-dashboard.md) or customize your settings with [Basic Configuration Guide](basic-configuration.md).*