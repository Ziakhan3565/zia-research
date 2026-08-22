# ========================================================
# HEADER
# ========================================================

if direction == "LONG":

    dir_color = "#00e676"

elif direction == "SHORT":

    dir_color = "#ff5252"

else:

    dir_color = "#38bdf8"


mins_rem, secs_rem = divmod(
    time_remaining,
    60
)


st.markdown(
    f"""
<div class="top-status-bar">

🟢 <b>{selected_symbol}</b>

&nbsp; | &nbsp;

Price:
<b>${close_p:,.2f}</b>

&nbsp; | &nbsp;

TF:
<b>{selected_tf_label}</b>

&nbsp; | &nbsp;

Signal:

<span style="color:{dir_color}; font-weight:800;">
{direction}
</span>

&nbsp; | &nbsp;

Score:
<b>{final_score:+.3f}</
