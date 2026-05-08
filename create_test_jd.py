import fitz

text = """Executive Assistant Position

We are seeking an exceptional Executive Assistant to support a founder, C-suite executive, or senior leader in a fast-paced tech environment.

Position Overview:
- Support founder/C-suite executive with calendar management, email handling, and administrative coordination
- Remote position with requirement for Pacific Time zone overlap (minimum 8am-2pm PST daily availability)
- Fast-paced startup environment requiring high initiative and proactive problem-solving

Key Responsibilities:
- Gmail mastery and calendar optimization
- Meeting scheduling and coordination across time zones
- Travel arrangements and logistics
- Drafting and editing communications
- Following up on action items proactively
- Confidentiality with sensitive information
- Managing multiple priorities simultaneously

Required Qualifications:
- 2+ years of experience supporting a founder, C-suite executive, or senior leader
- Exceptional written English (critical for email drafting)
- Gmail expert with advanced features knowledge
- Strong calendar management and organizational skills
- Proactive follow-up orientation and attention to detail
- Discretion with sensitive company and personal information
- AI-fluent (familiar with AI tools like ChatGPT, Claude for productivity)
"""

doc = fitz.open()
page = doc.new_page()
page.insert_text((50, 50), text, fontsize=11, fontname="helv")
doc.save("test_ea_jd.pdf")
print("Created test_ea_jd.pdf")
