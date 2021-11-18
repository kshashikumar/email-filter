from spamFilter import SpamClassifier, trainData, process_mails, metrics, testData
from flask import Flask, render_template, request, session, redirect
import smtplib
from email.message import EmailMessage
import imaplib     
import email 
import datetime
                  

server = smtplib.SMTP_SSL("smtp.gmail.com", 465)

username=""
pwd=""

sc = SpamClassifier(trainData)

app = Flask(__name__)
app.secret_key = "MailBoxProject"


@app.route('/')
def display_home():
    if 'user' in session:
        redirect("/logout")
    return redirect("/login")


@app.route('/login')
def display_login():
    return render_template('login.html')


@app.route('/home', methods=['GET', 'POST'])
def login():
    print("Request:", request.method)
    if request.method == 'GET':
        if 'user' in session:
            return render_template('compose.html', user=session['user'])
        else:
            return redirect("/login")
    else:
        det = request.form
        try:
            global username
            username=det.get('emailId')
            global pwd
            pwd=det.get('password')
            server.login(username,pwd)
            print(server.user)
        except:
            return render_template('signin.html', warn="Invalid credentials")

        session['user'] = det.get('emailId')
        return render_template('compose.html', user=session['user'])


@app.route('/result', methods=['POST'])
def display_result():
    mail = request.form.get('message')
    print(mail)
    pm = process_mails(mail)
    check = sc.classify(pm)
    print(check)
    if(check):
        res = "It is a Spam!"
    else:
        res = "It is not a Spam!"
    if 'user' in session:
        return render_template('result_login.html',result=res,msg=mail,user=session['user'])
    return render_template('result.html', result=res, msg=mail)


@app.route('/check')
def display_check():
    if 'user' in session:
        return redirect("/check_login")
    return render_template('check.html')


@app.route('/check_login')
def display_check_login():
    if 'user' not in session:
        return redirect("/check")
    return render_template('check_login.html', user=session['user'])

@app.route('/inbox')
def display_inbox():
    imap = imaplib.IMAP4_SSL("imap.gmail.com")
    imap.login(username, pwd)
    imap.list()
    imap.select('inbox')
    result, data = imap.uid('search', None, "ALL")  # (ALL/UNSEEN)
    i = len(data[0].split())
    spam_mails=[]
    ham_mails=[]
    body=""
    count=0
    for x in range(i-1,-1,-1):
        latest_email_uid = data[0].split()[x]
        result, email_data = imap.uid('fetch', latest_email_uid, '(RFC822)')
        raw_email = email_data[0][1]
        raw_email_string = raw_email.decode('utf-8')
        email_message = email.message_from_string(raw_email_string)
        if email_message.get_content_type() == "text/plain":
            body=email_message.get_payload(decode=True).decode()
            count+=1    
        elif email_message.get_content_type() == "multipart/alternative":     
                for part in email_message.get_payload()!=None and email_message.get_payload():
                    if part.get_content_type() == 'text/plain':
                        body = part.get_payload(decode=True).decode()
                count+=1      
        else:
            continue
        date_tuple = email.utils.parsedate_tz(email_message['Date'])
        if date_tuple:
            local_date = datetime.datetime.fromtimestamp(email.utils.mktime_tz(date_tuple))
            date = "%s" % (str(local_date.strftime("%a, %d %b %Y %H:%M:%S")))
        From=str(email.header.make_header(email.header.decode_header(email_message['From'])))
        sub=str(email.header.make_header(email.header.decode_header(email_message['Subject'])))
        pm=process_mails(body)
        check=sc.classify(pm)
        if check:
            spam_mails.append([date,From,sub,body])
        else:
            ham_mails.append([date,From,sub,body])
        if count == 10:
                break
    imap.logout()
    return render_template('inbox.html',user=session['user'] ,spamMails=spam_mails,hamMails=ham_mails,len_spam=len(spam_mails),len_ham=len(ham_mails))

@app.route('/send', methods=['POST'])
def display_alert():
    toMail = request.form.get('to')
    subject = request.form.get('sub')
    mail = request.form.get('mail')
    print(mail)
    pm = process_mails(mail)
    check = sc.classify(pm)
    print(check)
    if check:
        res = "Cannot send spam mails"
    else:
        fromMail = server.user
        mesg = EmailMessage()
        mesg.set_content(mail)
        mesg['Subject'] = subject
        mesg['From'] = fromMail
        mesg['To'] = toMail

        server.sendmail(fromMail, toMail, mesg.as_string())
        
        res = "Mail sent!"
    return render_template('sendMail.html', msg=res, user=server.user)


@app.route('/logout')
def logout():
    server.rset()
    session.clear()
    return redirect("/")

@app.route('/sentMails')
def sentMails():
    
    if 'user' not in session:
        return redirect("/login")
    
    imap = imaplib.IMAP4_SSL("imap.gmail.com")
    imap.login(username, pwd) 
    # select the e-mails
    res, messages = imap.select('"[Gmail]/Sent Mail"')   
    
    # calculates the total number of sent messages
    messages = int(messages[0])
    
    # determine the number of e-mails to be fetched
    n = 10
    
    result=[]
    # iterating over the e-mails
    for i in range(messages, messages - n, -1):
        res, msg = imap.fetch(str(i), "(RFC822)")     
        for response in msg:
            if isinstance(response, tuple):
                msg = email.message_from_bytes(response[1])


                To = msg["To"]
                # getting the subject of the sent mail
                subject = msg["Subject"]
                try:
                    body=msg.get_payload(decode=True).decode()
                except:
                    for part in msg.get_payload()!=None and msg.get_payload():
                        if part.get_content_type() == 'text/plain':
                            body = part.get_payload(decode=True).decode()
                result.append([To,subject,body])
    imap.logout()
    return render_template('sentMails.html',len=len(result),mails=result,user=session['user'])
       
if __name__ == "__main__":
    sc.train()
    preds_tf_idf = sc.predict(testData['message'])
    metrics(testData['label'], preds_tf_idf)
    app.run(debug=True)
