### ref-log.md

**What I learned**  
This assignment helped me understand how separating reasoning stages improves accuracy and realism. The Planner generates structured drafts without external data, while the Reviewer validates every claim using real-world information from online sources. I saw first-hand how this division mirrors real production pipelines where one system handles creative generation and another handles verification and grounding.

**Challenges**  
The main issue was ensuring the Reviewer didn’t rely on placeholder estimates. I solved this by redesigning its instructions so it must look up actual flight, lodging, food, and attraction prices from verified sources (Booking.com, Numbeo, official museum pages, etc.). I also ran into formatting issues with the dollar sign symbol causing Markdown to interpret parts of text as LaTeX math. I tried to fixed this by using Python raw strings (r"""...""") and explicitly telling the model to treat "$" as literal text. These changes made the output consistent and professional.

**Design choices**
I focused on realism, structure, and transparency. The Reviewer now outputs four standardized sections like Validation Log, Delta List, Final Revised Plan, and Trip Cost Summary with real data, proper evidence names, and strict formatting rules. I also enforced that all budgets include a 5–10% buffer, which improves credibility and user trust. Each itinerary now reports clear cost tables with supporting evidence sources and consistent currency notation.  

**External tools and GenAI usage**  
I used ChatGPT to polish prompt formatting and verify that my ideas and instructions are valid and would lead to good results. I made the instructions myself and mainly used AI for formatting in a way that OpenAI can get maximum value or understand it well.