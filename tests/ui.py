import asyncio
import json

import streamlit as st

from tests.add_multiply import brain

st.title("🧠 Brain Streaming Chat (ReAct)")

# --- 1) bootstrap history and a renderer ---
if "history" not in st.session_state:
    st.session_state.history = []  # list[dict]


def render_history():
    """Rebuild the UI from saved events."""
    for ev in st.session_state.history:
        kind = ev["kind"]
        if kind == "user":
            with st.chat_message("user"):
                st.markdown(ev["content"])
        elif kind == "assistant":
            with st.chat_message("assistant"):
                st.markdown(ev["content"])
        elif kind == "status":
            # Re-show a finished status block (collapsed or expanded as you like)
            with st.chat_message("assistant"):
                with st.status(f"{ev['state_name']} Done", expanded=False, state="complete"):
                    st.markdown(ev["content"])


render_history()

user_msg = st.chat_input("Ask me anything…")
if user_msg:
    st.session_state.history.append({"kind": "user", "content": user_msg})
    with st.chat_message("user"):
        st.markdown(user_msg)

    async def stream_response(user_msg=user_msg):
        state = None
        st_container = None
        slot = None
        buf = ""
        final_answer = ""

        async for raw in brain.answer(user_msg):
            chunk = json.loads(raw)
            curr_state = chunk.get("type", "")
            message = chunk.get("message", "")

            # When state changes, flush previous buffer to history and close UI
            if curr_state != state:
                if state is not None:
                    if state != "AnswerState":
                        st_container.update(label=f"{state} Done", state="complete", expanded=False)
                        st.session_state.history.append(
                            {
                                "kind": "status",
                                "state_name": state,
                                "content": buf,
                            }
                        )
                    else:
                        st.session_state.history.append(
                            {
                                "kind": "assistant",
                                "content": final_answer,
                            }
                        )

                # open a new container for the new state
                if curr_state == "AnswerState":
                    st_container = st.chat_message("assistant")
                    slot = st_container.empty()
                    final_answer = ""
                else:
                    with st.chat_message("assistant"):
                        st_container = st.status(curr_state, expanded=True, state="running")
                    slot = st_container.empty()
                    buf = ""

                state = curr_state

            # stream text into the current container
            if curr_state == "AnswerState":
                final_answer += message
                slot.markdown(final_answer)
            else:
                buf += message
                slot.markdown(buf)

        # stream ended → flush the last open block
        if state is not None:
            if state != "AnswerState":
                st_container.update(label=f"{state} Done", state="complete", expanded=True)
                st.session_state.history.append(
                    {
                        "kind": "status",
                        "state_name": state,
                        "content": buf,
                    }
                )
            else:
                st.session_state.history.append(
                    {
                        "kind": "assistant",
                        "content": final_answer,
                    }
                )

    asyncio.run(stream_response(user_msg=user_msg))
