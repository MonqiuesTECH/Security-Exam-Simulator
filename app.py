import streamlit as st

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        user = st.session_state["username"]
        # Fetch the password for the entered username from secrets
        if user in st.secrets["passwords"] and st.session_state["password"] == st.secrets["passwords"][user]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store password
            st.session_state["current_user"] = user # Save who logged in
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show inputs for username + password.
        st.title("🔒 Security+ Simulator Login")
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered)
        return False
    
    elif not st.session_state["password_correct"]:
        # Password not correct, show input + error.
        st.title("🔒 Security+ Simulator Login")
        st.text_input("Username", key="username")
        st.text_input("Password", type="password", key="password")
        st.button("Login", on_click=password_entered)
        st.error("😕 User not known or password incorrect")
        return False
    
    else:
        # Password correct.
        return True

# --- EXECUTING THE APP ---
if check_password():
    # If the user is logged in, show the sidebar logout button and run the app
    with st.sidebar:
        st.write(f"Logged in as: **{st.session_state.get('current_user', 'User')}**")
        if st.button("Log Out"):
            st.session_state.clear()
            st.rerun()
            
    # Run your main simulator function!
    # main()
