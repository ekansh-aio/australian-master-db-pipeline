"""
TARGETED FIXES for Observed Misclassifications

Based on error analysis showing:
1. legal_reasoning → legal_issues (reasoning seen as question-framing)
2. case_facts → procedural_history (facts confused with court steps)
3. decision → procedural_history (final orders confused with prior rulings)
4. arguments → legal_issues (submissions confused with issue statements)
5. legal_reasoning → decision (reasoning confused with conclusions)

STRATEGY: Add strong NEGATIVE contrasts and DISTINGUISHING markers
"""

ROLE_DESCRIPTIONS_DICT = {
    "procedural_history": [
        # CORE FUNCTION
        "Administrative steps and court management actions within the litigation process",
        "Timeline of formal filings and judicial case processing events",
        
        # SUPER DISTINCTIVE MARKERS (to prevent case_facts confusion)
        "Contains procedural verbs like: filed, commenced, listed, adjourned, heard, reserved, granted leave, refused, dismissed appeal",
        "Mentions court actions: writ filed, summons issued, matter listed, hearing adjourned, judgment reserved",
        "References to applications motions notices appeals being filed granted or refused",
        
        # TEMPORAL + PROCEDURAL COMBO
        "Dates combined with court actions: filed on DATE, listed for DATE, adjourned until DATE",
        "Sequential court events: first filed, then listed, subsequently adjourned, finally heard",
        
        # STRONG NEGATIVE BOUNDARIES
        "NOT real-world events like contracts accidents or business dealings",
        "NOT the court's final binding orders that end the case",
        "NOT the court's reasoning or interpretation of law",
        "Reports administrative scheduling and case management NOT substantive legal analysis",
        
        # EXAMPLES (court administration focus)
        "Proceedings were commenced by writ filed on 3 March 1991",
        "The matter was listed for directions on 15 July",
        "Leave to appeal was granted by the primary judge",
        "The hearing was adjourned pending further affidavits",
    ],


    "case_facts": [
        # CORE FUNCTION  
        "Real-world events and circumstances that occurred BEFORE and OUTSIDE the court process",
        "Background narrative of transactions relationships and conduct in the world not the courtroom",
        
        # SUPER DISTINCTIVE MARKERS (to prevent procedural_history confusion)
        "Describes actions in the real world: parties entered into, payment was made, accident occurred, business operated",
        "Past tense narrative of completed events: the plaintiff delivered, the defendant failed to pay, music was played",
        "Commercial or physical events: contracts signed, goods delivered, accidents happened, employees dismissed",
        "NO court actions: NO filing, NO listing, NO adjournment, NO hearing, NO applications",
        
        # CONTENT DISTINGUISHERS
        "Specific real-world details: amounts paid, dates of transactions, locations of events, names of parties to agreements",
        "Industry operations: how a business worked, licensing schemes, market conditions",
        "Relationships: employer-employee, landlord-tenant, supplier-customer",
        
        # STRONG NEGATIVE BOUNDARIES
        "NOT what happened inside the courtroom or registry",
        "NOT filings hearings applications or court procedural steps",
        "NOT what lawyers argued to the court",
        "NOT what the judge concluded",
        "If it would exist even without litigation it is case facts",
        
        # EXAMPLES (real world focus)
        "The parties entered into a lease agreement on 15 July 1989",
        "The plaintiff delivered 5000 units but payment was never received",
        "An accident occurred at the intersection of Smith and Jones Streets",
        "Music was played at the venue using pre-recorded backing tracks",
        "The employee was dismissed without notice on 20 June 1990",
    ],


    "legal_issues": [
        # CORE FUNCTION
        "Explicit framing of the legal question WITHOUT answering it",
        "Statement identifying what must be decided but NOT providing the answer",
        
        # SUPER DISTINCTIVE MARKERS (to prevent legal_reasoning and arguments confusion)
        "Begins with phrases: THE ISSUE IS WHETHER, THE QUESTION IS WHETHER, IT MUST BE DETERMINED WHETHER",
        "Uses question-framing not question-answering language",
        "Identifies the legal controversy but provides NO analysis or reasoning",
        "NO first-person judicial voice (I am satisfied, in my view)",
        "NO attribution to parties (counsel submitted, applicant argued)",
        
        # STRUCTURAL CHARACTERISTICS
        "Brief standalone statement usually early in judgment or at transitions",
        "Presents alternatives: whether X or Y, whether A applies or B applies",
        "Abstract legal question NOT concrete factual narrative",
        
        # STRONG NEGATIVE BOUNDARIES
        "NOT the court's reasoning or analysis (that's legal_reasoning)",
        "NOT a party's submission or argument (that's arguments)",
        "NOT the answer or conclusion (that's legal_reasoning or decision)",
        "ONLY identifies the question NEVER provides evaluation or determination",
        "If it contains because, therefore, accordingly, it follows = NOT legal_issues",
        "If attributed to a party = NOT legal_issues",
        
        # EXAMPLES (pure question-framing)
        "The issue is whether the Tribunal exceeded its jurisdiction",
        "The question is whether section 52 of the Act applies to this conduct",
        "Two issues arise: whether a duty of care existed and whether it was breached",
        "The central question is the proper construction of section 154",
        "It must be determined whether the contract is void for uncertainty",
    ],


    "arguments": [
        # CORE FUNCTION
        "Positions advanced BY PARTIES through their counsel NOT by the court",
        "What lawyers submitted contended or argued to persuade the judge",
        
        # SUPER DISTINCTIVE MARKERS (to prevent legal_issues confusion)
        "ALWAYS attributed to a party: the applicant submitted, counsel argued, the respondent contended, it was submitted that",
        "Contains attribution markers: submitted, argued, contended, maintained, advanced, put forward",
        "References counsel by role: senior counsel submitted, the appellant's counsel argued",
        "Advocatory persuasive tone: why the court SHOULD rule a certain way",
        "NO judicial voice: NEVER I am satisfied, in my view, we conclude",
        
        # CONTENT CHARACTERISTICS
        "Party's interpretation or construction of law",
        "Challenges to evidence or procedure by parties",
        "Policy reasons advanced by parties",
        "Requests for relief with party's supporting reasons",
        
        # STRONG NEGATIVE BOUNDARIES
        "NOT the court's own analysis or reasoning",
        "NOT neutral issue-framing by the court",
        "NOT the court's final determination",
        "MUST be attributed to a party NEVER to the judge",
        "If no attribution marker = probably NOT arguments",
        
        # EXAMPLES (clear attribution)
        "Senior counsel for the applicant submitted that section 52 should be read narrowly",
        "The respondent contended that the appeal should be dismissed with costs",
        "It was argued on behalf of the appellant that no duty of care arose",
        "The applicant maintained that the Tribunal lacked jurisdiction to make the order",
        "Counsel submitted that the evidence was inadmissible under section 138",
    ],


    "legal_reasoning": [
        # CORE FUNCTION
        "The COURT'S OWN analysis interpretation and application of law",
        "Judicial thinking and evaluation NOT party submissions",
        
        # SUPER DISTINCTIVE MARKERS (to prevent legal_issues and decision confusion)
        "First-person judicial voice: I am satisfied, I conclude, in my view, in my opinion, I find",
        "Plural judicial voice: we conclude, in our view, we accept, we reject",
        "Evaluative reasoning language: it follows that, therefore, accordingly, for these reasons, consequently",
        "Assessment of submissions: I accept that submission, I reject that contention, that argument must fail",
        "Application to facts: on these facts, in these circumstances, the evidence establishes",
        
        # ANALYTICAL OPERATIONS
        "Interpretation of statutes or precedents WITH explanation",
        "Weighing competing arguments and stating which is preferred and WHY",
        "Application of legal tests to facts with evaluative conclusion",
        "Explaining WHY a conclusion is reached NOT just stating it",
        
        # STRONG NEGATIVE BOUNDARIES
        "NOT just identifying the question (that's legal_issues)",
        "NOT attributed to parties (that's arguments)",
        "NOT the bare final order (that's decision)",
        "Contains explanation and reasoning NOT just conclusion",
        "If it's ONLY a question with no answer = legal_issues",
        "If it's ONLY a final order with no reasoning = decision",
        
        # EXAMPLES (judicial analysis)
        "Section 154 must be read in its broader statutory context including Part III",
        "I am not satisfied that the evidence establishes negligence on these facts",
        "For these reasons the submission that no duty arose must fail",
        "In my view the correct interpretation is that section 52 applies to this conduct",
        "The burden of proof lies on the applicant and has not been discharged on the evidence",
        "I accept the respondent's submission that the contract is void for uncertainty",
    ],


    "decision": [
        # CORE FUNCTION
        "Final operative orders that RESOLVE the dispute and bind parties",
        "The conclusive determination NOT the reasoning that led to it",
        
        # SUPER DISTINCTIVE MARKERS (to prevent procedural_history and legal_reasoning confusion)
        "Formal dispositive language: THE APPEAL IS DISMISSED, THE APPLICATION IS GRANTED, IT IS ORDERED THAT",
        "Present tense conclusive verbs: is dismissed, is allowed, is refused, is set aside, are affirmed",
        "Modal conclusions: I would dismiss, I would allow, I would set aside",
        "Cost orders: costs are awarded, no order as to costs, respondent must pay costs",
        "Numbered formal orders usually at the END of judgment",
        
        # DISTINGUISHING FROM LEGAL_REASONING
        "Operative command or determination NOT explanation of reasoning",
        "Can include transitional phrase (for these reasons) but MUST contain final operative order",
        "States WHAT the court orders NOT WHY it orders it",
        
        # DISTINGUISHING FROM PROCEDURAL_HISTORY
        "THIS COURT's final binding orders NOT reports of what a lower court did",
        "If reporting prior court ruling = procedural_history",
        "If THIS court's final determination = decision",
        
        # STRONG NEGATIVE BOUNDARIES
        "NOT explanation of legal reasoning (that's legal_reasoning)",
        "NOT reporting what a lower court ordered (that's procedural_history)",
        "NOT interlocutory or procedural directions (that's procedural_history)",
        "MUST be final and dispositive",
        
        # EXAMPLES (final orders)
        "The appeal is dismissed with costs",
        "I would set aside the decision of the Tribunal and remit for reconsideration",
        "Order that the respondent pay the applicant's costs as agreed or assessed",
        "For these reasons the appeal must be dismissed",
        "Judgment for the defendant with costs",
        "The application is refused",
    ],


    "other": [
        # FUNCTION
        "Non-substantive administrative metadata formatting or publication information",
        
        # CONTENT TYPES
        "Page numbers headers footers running heads",
        "Court file numbers case citations",
        "Publication metadata copyright notices download information",
        "Purely structural elements with no legal content",
        
        # EXAMPLES
        "Page 47 of 92",
        "IN THE HIGH COURT OF AUSTRALIA",
        "No 1234 of 2020",
        "Downloaded from www.austlii.edu.au",
        "Copyright Commonwealth of Australia",
        
        # CHARACTERISTICS
        "No substantive legal content",
        "Could be removed without affecting understanding",
    ]
}


# =============================================================================
# ANALYSIS OF FIXES
# =============================================================================

TARGETED_FIXES_APPLIED = """
MISCLASSIFICATION: legal_reasoning → legal_issues
ROOT CAUSE: Reasoning chunks that identify issues were matching issue-framing language

FIXES APPLIED:
1. Added to legal_issues:
   - "Identifies the legal controversy but provides NO analysis or reasoning"
   - "NO first-person judicial voice (I am satisfied, in my view)"
   - "ONLY identifies the question NEVER provides evaluation"
   - "If it contains because, therefore, accordingly = NOT legal_issues"

2. Added to legal_reasoning:
   - "Evaluative reasoning language: it follows that, therefore, accordingly"
   - "Contains explanation and reasoning NOT just conclusion"
   - "If it's ONLY a question with no answer = legal_issues"

---

MISCLASSIFICATION: case_facts → procedural_history
ROOT CAUSE: Factual narratives matched procedural language or dates

FIXES APPLIED:
1. Added to case_facts:
   - "Describes actions in the real world: parties entered into, payment was made"
   - "NO court actions: NO filing, NO listing, NO adjournment, NO hearing"
   - "If it would exist even without litigation it is case facts"

2. Added to procedural_history:
   - "Contains procedural verbs like: filed, commenced, listed, adjourned"
   - "NOT real-world events like contracts accidents or business dealings"
   - "Reports administrative scheduling NOT substantive events"

---

MISCLASSIFICATION: decision → procedural_history
ROOT CAUSE: Final orders confused with reports of prior court rulings

FIXES APPLIED:
1. Added to decision:
   - "THIS COURT's final binding orders NOT reports of what a lower court did"
   - "If reporting prior court ruling = procedural_history"
   - "If THIS court's final determination = decision"

2. Added to procedural_history:
   - "NOT the court's final binding orders that end the case"
   - Clarified it reports prior steps not final determinations

---

MISCLASSIFICATION: arguments → legal_issues
ROOT CAUSE: Party submissions about issues matched issue-framing language

FIXES APPLIED:
1. Added to arguments:
   - "ALWAYS attributed to a party: the applicant submitted, counsel argued"
   - "MUST be attributed to a party NEVER to the judge"
   - "If no attribution marker = probably NOT arguments"

2. Added to legal_issues:
   - "NO attribution to parties (counsel submitted, applicant argued)"
   - "If attributed to a party = NOT legal_issues"

---

MISCLASSIFICATION: legal_reasoning → decision
ROOT CAUSE: Reasoning with conclusory language matched decision patterns

FIXES APPLIED:
1. Added to legal_reasoning:
   - "Contains explanation and reasoning NOT just conclusion"
   - "NOT the bare final order (that's decision)"
   - "Explaining WHY a conclusion is reached NOT just stating it"

2. Added to decision:
   - "Operative command or determination NOT explanation of reasoning"
   - "States WHAT the court orders NOT WHY it orders it"
   - "NOT explanation of legal reasoning (that's legal_reasoning)"
"""


if __name__ == "__main__":
    print("="*80)
    print("REFINED ROLE DESCRIPTIONS - TARGETED FIXES")
    print("="*80)
    print("\nBased on observed misclassifications:")
    print("1. legal_reasoning → legal_issues")
    print("2. case_facts → procedural_history")
    print("3. decision → procedural_history")
    print("4. arguments → legal_issues")
    print("5. legal_reasoning → decision")
    print("\n" + TARGETED_FIXES_APPLIED)
    
    print("\n" + "="*80)
    print("USAGE")
    print("="*80)
    print("""
from refined_role_descriptions import REFINED_ROLE_DESCRIPTIONS
from improved_role_classifier import ImprovedEmbeddingRoleClassifier

classifier = ImprovedEmbeddingRoleClassifier(
    role_descriptions=REFINED_ROLE_DESCRIPTIONS,
    aggregation_method='top_k_mean',
    top_k_descriptions=3,
    confidence_threshold=0.30
)

# Re-test on your data
results = classifier.classify_chunks(your_chunks)
    """)