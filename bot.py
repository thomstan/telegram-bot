import os, json
from telegram import Update, BotCommand,InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler,
    MessageHandler, filters, ContextTypes, CallbackQueryHandler
)
USERS_FILE = "data/user.json"
ANNOUNCE_FILE = "data/announcement.txt"
import os, json, uuid
from mycombined import extract_right_side_fields, place_on_template
from dotenv import load_dotenv
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
ADMIN_ID = int(os.getenv(ADMIN_ID))  # your Telegram ID

DB_FILE = "payments.json"
def save_user(user_id):
    os.makedirs("data", exist_ok=True)

    users = set()
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE) as f:
            users = set(json.load(f))

    users.add(int(user_id))

    with open(USERS_FILE, "w") as f:
        json.dump(list(users), f)


def get_announcement():
    if os.path.exists(ANNOUNCE_FILE):
        with open(ANNOUNCE_FILE, encoding="utf-8") as f:
            return f.read().strip()
    return None

def load_db():
    return json.load(open(DB_FILE)) if os.path.exists(DB_FILE) else {}

def save_db(db):
    json.dump(db, open(DB_FILE, "w"), indent=2)

# ---------------- START ----------------

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_name = update.effective_user.first_name

    # 1️⃣ Set bot commands (menu under typing bar)
    commands = [
        BotCommand("usage", "መመሪያ / How to use the bot"),
        BotCommand("payment", "ክፍያ / Payment info"),
        BotCommand("help", "እርዳታ / Help"),
    ]
    await context.bot.set_my_commands(commands)

    # 2️⃣ Send combined greeting + instructions
    text = (
        f"👋 Hello {user_name}!\n"
        "Welcome to FAYDA PRINT Bot.\n\n"
        "📌 Commands are available under the typing bar. Tap them to interact.\n\n"
        "Available commands:\n"
        "/usage - መመሪያ / How to use the bot\n"
        "/payment - ክፍያ / Payment info\n"
        "/help - እርዳታ / Help\n\n"
        "Follow the instructions carefully to send your documents and receive approval."
    )

    await update.message.reply_text(text)

# ---------------- PDF HANDLER ----------------
async def handle_pdf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    save_user(update.effective_user.id)

    announcement = get_announcement()
    if announcement:
      await update.message.reply_text(
        f"📢 *Announcement*\n\n{announcement}",
        parse_mode="Markdown"
    )
    await update.message.reply_text(
        "📥 Receiving your PDF...\n⏳ Please wait."
    )

    if update.message.document.file_size > 10 * 1024 * 1024:
        await update.message.reply_text(
            "❌ File too large. Max allowed is 10 MB."
        )
        return

    user_id = str(update.message.from_user.id)
    file = await update.message.document.get_file()

    os.makedirs("uploads", exist_ok=True)
    pdf_path = f"uploads/{uuid.uuid4().hex}.pdf"

    try:
        await file.download_to_drive(pdf_path)
    except Exception as e:
        await update.message.reply_text(
            "❌ Download failed. Please try again."
        )
        print("DOWNLOAD ERROR:", e)
        return

    await update.message.reply_text("⚙️ Processing document...")

    results = extract_right_side_fields(pdf_path)

    os.makedirs("out_right/previews", exist_ok=True)
    os.makedirs("out_right/finals", exist_ok=True)

    preview_path = f"out_right/previews/{user_id}.png"
    final_path   = f"out_right/finals/{user_id}.png"

    place_on_template(pdf_path, results, preview_path, watermark=True)
    place_on_template(pdf_path, results, final_path, watermark=False)

    from telegram import InlineKeyboardButton, InlineKeyboardMarkup

    keyboard = InlineKeyboardMarkup([
      [
        InlineKeyboardButton("✅ Approve", callback_data=f"approve:{user_id}"),
        InlineKeyboardButton("❌ Reject", callback_data=f"reject:{user_id}")
       ]
    ])
    

    with open(preview_path, "rb") as photo:
        await update.message.reply_photo(
           photo=photo,
           caption="👁 Watermarked preview. Pay 100 birr to receive final ID.\n💳 Telebirr / CBE Birr: +251992181173"
        )

    with open(preview_path, "rb") as photo:
        await context.bot.send_photo(
           chat_id=ADMIN_ID,
           photo=photo,
           caption=f"👁 Preview\nUser ID: {user_id}\nApprove?",
           reply_markup=keyboard
        )

async def usage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "📌 እንዴት ቦትን መጠቀም እንደሚቻል:\n"
        "የFAYDA መተግበሪያ ያውርዱ በመቀጠልም የመተግበሪያው መመሪያ በመከተል መረጃዎን(printable credential PDF) ካወረዱ በኋላ ከታች ያሉተን በቅደም ተከተል ይተግብሩ።\n"
        "1️⃣ ፒዲኤፍ ይላኩ\n"
        "2️⃣ ቦቱ እስከሚሠራበት ይጠብቁ\n"
        "3️⃣ ክፍያ ከተከፈለ በኋላ አስተዳዳሪው ይፀድቃል\n\n"
        "How to use the bot:\n"
        "1️⃣ Send your PDF document\n"
        "2️⃣ Wait for processing\n"
        "3️⃣ Pay via Telebirr or CBE Birr"
    )
    await update.message.reply_text(text)


async def payment(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "💳 ክፍያ መረጃ / Payment info:\n"
        "Telebirr: +251992181173\n"
        "CBE Birr: +251992181173\n\n"
        "💡 After payment, the admin will approve your document."
    )
    await update.message.reply_text(text)


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = (
        "❓ እርዳታ / Help:\n"
        "Use /usage to see how to send your PDF.\n"
        "Use /payment to see payment info.\n"
        "Admin will approve your document after payment."
    )
    await update.message.reply_text(text)

# ---------------- PAYMENT PROOF ----------------
async def handle_payment_proof(update: Update, context: ContextTypes.DEFAULT_TYPE):
    save_user(update.effective_user.id)
    announcement = get_announcement()
    if announcement:
       await update.message.reply_text(
        f"📢 *Announcement*\n\n{announcement}",
        parse_mode="Markdown"
    )
    user_id = str(update.message.from_user.id)
    db = load_db()

    if user_id not in db:
        await update.message.reply_text("❌ No pending request found.")
        return

    db[user_id]["status"] = "WAITING_ADMIN_APPROVAL"
    save_db(db)

    await update.message.reply_text("⏳ Payment received. Waiting for admin approval.")

    # Notify admin
    await context.bot.send_message(
    ADMIN_ID,
    f"💳 PAYMENT PROOF RECEIVED\n\n"
    f"👤 User ID: {user_id}\n\n"
    f"Approve with:\n"
    f"/approve {user_id}"
)
    if update.message.photo:
        await context.bot.send_photo(ADMIN_ID, update.message.photo[-1].file_id)
    else:
        await context.bot.send_message(ADMIN_ID, update.message.text)

# ---------------- ADMIN APPROVE ----------------
async def approve(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.from_user.id != ADMIN_ID:
        await update.message.reply_text("❌ You are not authorized.")
        return

    if not context.args:
        await update.message.reply_text(
            "❌ Missing user ID.\n\n"
            "Usage:\n/approve <USER_ID>"
        )
        return

    user_id = context.args[0]
    db = load_db()

    if user_id not in db:
        await update.message.reply_text("❌ User ID not found.")
        return

    await context.bot.send_photo(
        chat_id=int(user_id),
        photo=open(db[user_id]["final"], "rb"),
        caption="✅ Payment approved. Here is your final document."
    )

    db[user_id]["status"] = "COMPLETED"
    save_db(db)

    await update.message.reply_text(f"✅ Approved and sent to user {user_id}")

async def announce(update: Update, context: ContextTypes.DEFAULT_TYPE):
    save_user(update.effective_user.id)

    if update.effective_user.id !=  ADMIN_ID:
        await update.message.reply_text("❌ Admin only.")
        return

    message = " ".join(context.args)
    if not message:
        await update.message.reply_text("Usage: /announce <message>")
        return

    os.makedirs("data", exist_ok=True)
    with open(ANNOUNCE_FILE, "w", encoding="utf-8") as f:
        f.write(message)

    await update.message.reply_text("✅ Announcement saved.")

async def broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.effective_user.id != ADMIN_ID:
        return

    message = " ".join(context.args)
    if not message:
        await update.message.reply_text("Usage: /broadcast <message>")
        return

    with open(USERS_FILE) as f:
        users = json.load(f)

    sent = 0
    for uid in users:
        try:
            await context.bot.send_message(
                chat_id=uid,
                text=f"📢 Announcement\n\n{message}"
            )
            sent += 1
        except:
            pass

    await update.message.reply_text(f"✅ Sent to {sent} users.")

async def handle_approval(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    admin_id = query.from_user.id
    if admin_id != ADMIN_ID:
        await query.edit_message_caption("unothorized usage")
        return

    action, user_id = query.data.split(":")
    user_id = int(user_id)

    final_path = f"out_right/finals/{user_id}.png"

    if not os.path.exists(final_path):
        await query.edit_message_caption("❌ Final file not found.")
        return

    if action == "approve":
        await context.bot.send_photo(
            chat_id=user_id,
            photo=open(final_path, "rb"),
            caption="✅ Approved! Here is your final document."
        )
        await query.edit_message_caption("✅ Approved and sent.")

    elif action == "reject":
        await context.bot.send_message(
            chat_id=user_id,
            text="❌ Rejected. Please re-upload your document."
        )
        await query.edit_message_caption("❌ Rejected.")
    print("BUTTON CLICKED:", query.data)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    save_user(update.effective_user.id)

    announcement = get_announcement()
    if announcement:
        await update.message.reply_text(
            f"📢 Announcement\n\n{announcement}"
        )
        return

    await update.message.reply_text(
        "🤖 Please send a PDF document to begin."
    )

# ---------------- MAIN ----------------
from telegram.request import HTTPXRequest

request = HTTPXRequest(
    connect_timeout=30,
    read_timeout=300,   # VERY IMPORTANT for PDFs
    write_timeout=300,
    pool_timeout=300
)

app = ApplicationBuilder() \
    .token(BOT_TOKEN) \
    .request(request) \
    .build()

# Commands
app.add_handler(CommandHandler("start", start))
app.add_handler(CommandHandler("usage", usage))
app.add_handler(CommandHandler("payment", payment))
app.add_handler(CommandHandler("help", help_cmd))
app.add_handler(CommandHandler("announce", announce))
app.add_handler(CommandHandler("broadcast", broadcast))
#app.add_handler(CommandHandler("approve", approve))

# Callback buttons
app.add_handler(
    CallbackQueryHandler(handle_approval, pattern="^(approve|reject):")
)

# Documents
app.add_handler(MessageHandler(filters.Document.PDF, handle_pdf))

# Payment proof (PHOTO or TEXT like transaction ID)
app.add_handler(
    MessageHandler(
        filters.PHOTO | (filters.TEXT & ~filters.COMMAND),
        handle_payment_proof
    )
)

# General text (LAST)
app.add_handler(
    MessageHandler(
        filters.TEXT & ~filters.COMMAND,
        handle_text
    )
)

app.run_polling()
