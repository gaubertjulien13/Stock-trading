# test_email_alerts.py
import os
from dotenv import load_dotenv
import smtplib
from email.message import EmailMessage
from textwrap import dedent

# Load environment
load_dotenv('.stock_screener.env')

def send_email_alert(smtp_host, smtp_port, smtp_user, smtp_pass, to_list, subject, body):
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = smtp_user if smtp_user else "alerts@localhost"
    msg["To"] = ", ".join(to_list)
    msg.set_content(body)

    if smtp_port == 465:
        with smtplib.SMTP_SSL(smtp_host, smtp_port) as s:
            if smtp_user and smtp_pass:
                s.login(smtp_user, smtp_pass)
            s.send_message(msg)
    else:
        with smtplib.SMTP(smtp_host, smtp_port) as s:
            s.ehlo()
            try:
                s.starttls()
                s.ehlo()
            except Exception:
                pass
            if smtp_user and smtp_pass:
                s.login(smtp_user, smtp_pass)
            s.send_message(msg)

def test_alert():
    # Get credentials
    smtp_user = os.environ.get("ALERT_FROM_EMAIL")
    smtp_pass = os.environ.get("SMTP_APP_PASSWORD")
    to_list = [smtp_user]  # Send to yourself
    
    # Simulate alert data
    tkr = "AAPL"
    company_name = "Apple Inc."
    sector = "Technology"
    price = 185.50
    rsi = 65.5
    ema_s = 2.34
    sma_s = 1.85
    
    subj = f"🚨 TEST ALERT: {tkr} {company_name} — 15m"
    body = dedent(f"""\
        **THIS IS A TEST ALERT**
        
        Strict multi-method BUY signal detected!

        Ticker:       {tkr}
        Company:      {company_name}
        Sector:       {sector}
        Interval:     15m
        Price:        ${price:.2f}
        RSI:          {rsi:.1f}
        EMA strength: {ema_s:.2f}%
        SMA strength: {sma_s:.2f}%

        Conditions met:
          ✅ All methodologies fired (SMA + EMA/RSI + Bollinger)
          ✅ Above SMA200 trend filter
          ✅ MACD bullish confirmation
          
        This is a TEST alert to verify the system works.
    """)
    
    try:
        send_email_alert(
            smtp_host="smtp.gmail.com",
            smtp_port=465,
            smtp_user=smtp_user,
            smtp_pass=smtp_pass,
            to_list=to_list,
            subject=subj,
            body=body
        )
        print("✅ Test alert sent successfully!")
    except Exception as e:
        print(f"❌ Test alert failed: {e}")

if __name__ == "__main__":
    test_alert()