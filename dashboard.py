# ==========================================================
# PROFESSIONAL SIGNAL ENGINE
# LONG / SHORT / STRONG LONG / STRONG SHORT / WAIT
# ==========================================================

# ----------------------------------------------------------
# 1. XGB SIGNED SCORE
# ----------------------------------------------------------
if xgb_signal == "LONG":
    xgb_signed = xgb_confidence / 100.0
elif xgb_signal == "SHORT":
    xgb_signed = -(xgb_confidence / 100.0)
else:
    xgb_signed = 0.0


# ----------------------------------------------------------
# 2. MARKET COMPONENT SCORES
# ----------------------------------------------------------

# Research
research_direction = (
    "LONG" if research_score >= 0.10
    else "SHORT" if research_score <= -0.10
    else "NEUTRAL"
)

# Microstructure
micro_score = float(np.clip(
    0.65 * obi_val + 0.35 * ofi_norm,
    -1.0,
    1.0
))

micro_direction = (
    "LONG" if micro_score >= 0.08
    else "SHORT" if micro_score <= -0.08
    else "NEUTRAL"
)

# Trend
trend_direction = (
    "LONG" if trend_score >= 0.10
    else "SHORT" if trend_score <= -0.10
    else "NEUTRAL"
)


# ----------------------------------------------------------
# 3. DIRECTION VOTES
# ----------------------------------------------------------

signals = [
    xgb_signal,
    research_direction,
    micro_direction,
    trend_direction
]

long_votes = signals.count("LONG")
short_votes = signals.count("SHORT")


# ----------------------------------------------------------
# 4. DIRECTIONAL SCORE
# ----------------------------------------------------------

combined_score = float(np.clip(
    0.40 * xgb_signed +
    0.25 * research_score +
    0.20 * micro_score +
    0.15 * trend_score,
    -1.0,
    1.0
))


# ----------------------------------------------------------
# 5. XGB QUALITY
# ----------------------------------------------------------

xgb_strong = xgb_confidence >= 75.0
xgb_good = xgb_confidence >= 60.0


# ----------------------------------------------------------
# 6. OBI QUALITY
# ----------------------------------------------------------

obi_long = obi_val >= 0.10
obi_short = obi_val <= -0.10

strong_obi_long = obi_val >= 0.20
strong_obi_short = obi_val <= -0.20


# ----------------------------------------------------------
# 7. OFI QUALITY
# ----------------------------------------------------------

ofi_long = ofi_norm >= 0.05
ofi_short = ofi_norm <= -0.05

strong_ofi_long = ofi_norm >= 0.12
strong_ofi_short = ofi_norm <= -0.12


# ----------------------------------------------------------
# 8. MOMENTUM
# ----------------------------------------------------------

momentum_long = momentum5 > 0
momentum_short = momentum5 < 0


# ----------------------------------------------------------
# 9. NORMAL SIGNAL CONDITIONS
# ----------------------------------------------------------

long_normal = (
    long_votes >= 3
    and xgb_good
    and combined_score >= 0.12
    and trend_direction != "SHORT"
)

short_normal = (
    short_votes >= 3
    and xgb_good
    and combined_score <= -0.12
    and trend_direction != "LONG"
)


# ----------------------------------------------------------
# 10. STRONG LONG
# ----------------------------------------------------------

strong_long = (
    xgb_signal == "LONG"
    and xgb_strong
    and long_votes >= 3
    and combined_score >= 0.28
    and trend_direction == "LONG"
    and research_direction == "LONG"
    and micro_direction == "LONG"
    and (strong_obi_long or strong_ofi_long)
)


# ----------------------------------------------------------
# 11. STRONG SHORT
# ----------------------------------------------------------

strong_short = (
    xgb_signal == "SHORT"
    and xgb_strong
    and short_votes >= 3
    and combined_score <= -0.28
    and trend_direction == "SHORT"
    and research_direction == "SHORT"
    and micro_direction == "SHORT"
    and (strong_obi_short or strong_ofi_short)
)


# ----------------------------------------------------------
# 12. FINAL DIRECTION
# ----------------------------------------------------------

if strong_long:

    direction = "LONG"
    signal_strength = "STRONG LONG"

elif strong_short:

    direction = "SHORT"
    signal_strength = "STRONG SHORT"

elif long_normal:

    direction = "LONG"
    signal_strength = "LONG"

elif short_normal:

    direction = "SHORT"
    signal_strength = "SHORT"

else:

    direction = "NEUTRAL"
    signal_strength = "WAIT"


# ----------------------------------------------------------
# 13. FINAL CONFIDENCE
# ----------------------------------------------------------

base_confidence = abs(combined_score) * 100.0

# Agreement bonus
if direction == "LONG":
    agreement_bonus = long_votes * 4.0
elif direction == "SHORT":
    agreement_bonus = short_votes * 4.0
else:
    agreement_bonus = 0.0

# XGB bonus
xgb_bonus = max(0.0, (xgb_confidence - 50.0) * 0.15)

confidence = int(np.clip(
    base_confidence +
    agreement_bonus +
    xgb_bonus,
    0,
    99
))


# ----------------------------------------------------------
# 14. CONFIDENCE QUALITY FILTER
# ----------------------------------------------------------

# Weak LONG/SHORT ko WAIT mein convert karo
if direction == "LONG" and confidence < 58:
    direction = "NEUTRAL"
    signal_strength = "WAIT"

elif direction == "SHORT" and confidence < 58:
    direction = "NEUTRAL"
    signal_strength = "WAIT"


# ----------------------------------------------------------
# 15. FINAL SAFETY CHECK
# ----------------------------------------------------------

# Opposite strong trend ko allow nahi karna
if direction == "LONG" and trend_direction == "SHORT":
    direction = "NEUTRAL"
    signal_strength = "WAIT"

elif direction == "SHORT" and trend_direction == "LONG":
    direction = "NEUTRAL"
    signal_strength = "WAIT"


# ----------------------------------------------------------
# 16. VERY STRONG CONFLUENCE UPGRADE
# ----------------------------------------------------------

if direction == "LONG":

    if (
        xgb_confidence >= 80
        and long_votes >= 4
        and combined_score >= 0.35
        and obi_val >= 0.15
        and trend_direction == "LONG"
    ):
        signal_strength = "STRONG LONG"


elif direction == "SHORT":

    if (
        xgb_confidence >= 80
        and short_votes >= 4
        and combined_score <= -0.35
        and obi_val <= -0.15
        and trend_direction == "SHORT"
    ):
        signal_strength = "STRONG SHORT"


# ----------------------------------------------------------
# 17. FINAL WAIT RULE
# ----------------------------------------------------------

if direction == "NEUTRAL":
    signal_strength = "WAIT"
