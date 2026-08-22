# ==========================================
# 7. TOP HEADER STATUS BAR
# ==========================================

dir_color = (
    "#00e676" if direction == "LONG"
    else "#ff5252" if direction == "SHORT"
    else "#38bdf8"
)

mins_rem, secs_rem = divmod(time_remaining, 60)

# ==========================================
# DYNAMIC RISK / REWARD CALCULATION
# ==========================================

if direction == "LONG":
    risk_distance = abs(close_p - sl_val)
    reward_distance = abs(tp2_val - close_p)

elif direction == "SHORT":
    risk_distance = abs(sl_val - close_p)
    reward_distance = abs(close_p - tp2_val)

else:
    risk_distance = 0.0
    reward_distance = 0.0

if direction != "NEUTRAL" and risk_distance > 0:
    rr_ratio = reward_distance / risk_distance
    rr_display = f"1 : {rr_ratio:.2f}"
else:
    rr_ratio = 0.0
    rr_display = "—"


# ==========================================
# TOP HORIZONTAL STATUS BAR
# ==========================================

st.markdown(
    f"""
    <div class="top-status-bar" style="
        display:flex;
        align-items:center;
        justify-content:center;
        gap:14px;
        white-space:nowrap;
        overflow:hidden;
        font-size:13px;
        padding:12px 16px;
    ">
        <span>
            🟢 <b>{selected_symbol}</b>
        </span>

        <span>|</span>

        <span>
            Price: <b>${close_p:,.2f}</b>
        </span>

        <span>|</span>

        <span>
            TF: <b>{selected_tf_label}</b>
        </span>

        <span>|</span>

        <span>
            Signal:
            <b style="color:{dir_color};">
                {direction}
            </b>
        </span>

        <span>|</span>

        <span>
            Score:
            <b>{final_score:+.3f}</b>
        </span>

        <span>|</span>

        <span>
            Confidence:
            <b>{confidence}%</b>
        </span>

        <span>|</span>

        <span>
            RR:
            <b style="color:#38bdf8;">
                {rr_display}
            </b>
        </span>

        <span>|</span>

        <span>
            Next Reset:
            <b>{mins_rem}m {secs_rem}s</b>
        </span>
    </div>
    """,
    unsafe_allow_html=True
)
