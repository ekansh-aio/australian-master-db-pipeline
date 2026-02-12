ROLE_DESCRIPTIONS_DICT = {

"procedural_history": [

    # FUNCTION
    "Narrative of steps taken within this specific litigation",
    "Chronological account of how the case progressed in court or tribunal",
    "Description of filings hearings applications and procedural management",

    # TYPICAL CONTENT EXPANDED
    "Commencement of proceedings writ originating motion or reference",
    "Appeals being filed granted refused dismissed or withdrawn",
    "Applications for stay injunction summary judgment or security for costs",
    "Affidavits evidence or submissions being filed or tendered",
    "Hearings listings adjournments and directions",
    "Objections taken during proceedings",
    "Interlocutory rulings not determining final merits",
    "Leave granted or refused before final determination",
    "Relief sought prior to judgment",
    "Prior decisions of lower courts in the same matter",

    # STRONG EXAMPLES
    "Proceedings were commenced by writ filed on 3 March 1991",
    "The applicant seeks a declaration that the decision be set aside",
    "An application for a stay was made pending appeal",
    "Affidavits were filed by both parties",
    "The matter was listed for directions",
    "Submissions were heard over two days",
    "The Full Court allowed the appeal and remitted the matter",
    "The hearing was adjourned until 10.40 am",
    "Security for costs was sought by the respondent",
    "The proceedings were commenced by originating summons",
    "Leave to appeal was granted by the primary judge",
    "Directions were made for the filing of further evidence",
    "The respondent filed a notice of contention",
    "The matter was reserved for judgment",

    # EDGE CASE CLARIFICATION
    "Includes transcript management such as adjournments objections and case control",
    "Includes description of procedural posture at time of judgment",
    "Includes references brought under statutory provisions as procedural acts",
    "Statements describing steps taken by the court default to procedural history",
    "Reporting what a lower court decided in the same litigation is procedural history",

    # NEGATIVE BOUNDARY
    "Does not describe events occurring outside the litigation",
    "Does not evaluate legal arguments",
    "Does not interpret statutory language in detail",
    "Does not contain the final operative orders",
    "Does not consist solely of administrative headings",
    "Judicial evaluation of whether leave should have been granted is not procedural history"
],


"case_facts": [

    # FUNCTION
    "Description of events circumstances and conduct occurring outside the court process",
    "Narrative of transactions relationships and real world events giving rise to dispute",
    "Contextual background necessary to understand the controversy",

    # TYPICAL CONTENT EXPANDED
    "Contracts agreements leases or commercial arrangements",
    "Accidents incidents injuries or property damage",
    "Financial transactions loans transfers or unpaid sums",
    "Operation of a business industry or licensing scheme",
    "Employment relationships dismissal or workplace conduct",
    "Pre litigation negotiations or correspondence",
    "Membership structure of associations or corporate entities",
    "Industry practice or market conditions",
    "Historical development of a policy scheme or enterprise",
    "Evidence summaries describing factual events",

    # STRONG EXAMPLES
    "The parties entered into a lease in 1989",
    "The plaintiff delivered the goods but payment was not made",
    "An accident occurred at the intersection",
    "Music was played using recorded tracks at the venue",
    "Members of the applicant constituted approximately 8500 composers",
    "The licence fee was calculated by reference to venue capacity",
    "The employee was dismissed without notice",
    "A letter of demand was sent prior to proceedings",
    "The agreement was executed on 12 July 2004",
    "The plaintiff transferred 50000 dollars to the defendant",
    "The respondent failed to comply with safety regulations",
    "The parties exchanged correspondence over several months",

    # EDGE CASE CLARIFICATION
    "Includes summaries of affidavit evidence describing real world events",
    "Includes commercial structure underlying dispute",
    "Includes description of statutory scheme only insofar as it explains industry operation not interpretation",
    "If events described would exist even without litigation it is case facts",
    "Chronological narration of events leading to dispute is case facts",

    # NEGATIVE BOUNDARY
    "Does not describe filings hearings or court management",
    "Does not frame the legal question to be decided",
    "Does not interpret statutory provisions or legal doctrine",
    "Does not announce the court's decision",
    "Does not consist solely of metadata or headings",
    "Judicial findings about whether a contract was void are not case facts"
],


"legal_issues": [

    # FUNCTION
    "Explicit identification of the legal question requiring resolution",
    "Framing of the dispute in legal terms",
    "Definition of the point the court must determine",

    # TYPICAL CONTENT EXPANDED
    "Statements beginning with the issue is whether",
    "Statements beginning with the question is whether",
    "Identification of competing interpretations of statute",
    "Jurisdictional challenges",
    "Contested liability or enforceability questions",
    "Scope of statutory power or discretion",
    "Validity of administrative action",
    "Availability of remedies such as damages injunction or stay",

    # STRONG EXAMPLES
    "The issue is whether the contract is enforceable",
    "The question is whether the Tribunal exceeded its jurisdiction",
    "It must be determined whether a duty of care arose",
    "The dispute concerns whether damages are recoverable",
    "The central question is the proper construction of section 154",
    "The issue arises as to whether leave should be granted",
    "The appeal raises the question of statutory interpretation",
    "Two issues arise for consideration",

    # EDGE CASE CLARIFICATION
    "Includes statements identifying the legal controversy even if introduced through party contentions",
    "Includes transitional paragraphs narrowing the scope of determination",
    "If the paragraph frames what must be decided but does not resolve it it is legal issues",

    # NEGATIVE BOUNDARY
    "Does not provide extended reasoning or doctrinal analysis",
    "Does not merely recount factual background",
    "Does not describe procedural chronology",
    "Does not declare the final orders",
    "Does not assess evidence in detail",
    "Immediate resolution of the question moves into legal reasoning"
],


"arguments": [

    # FUNCTION
    "Submissions advanced by parties in support of their position",
    "Contentions and reasoning attributed to counsel or litigants",

    # TYPICAL CONTENT EXPANDED
    "Statements beginning with it was submitted that",
    "Counsel argued or contended that",
    "Objections raised to evidence or procedure",
    "Policy arguments advanced by parties",
    "Competing constructions proposed by parties",
    "Requests for specific remedies supported by reasoning",

    # STRONG EXAMPLES
    "The applicant submitted that the statutory language was ambiguous",
    "The respondent contended that the appeal should be dismissed",
    "Counsel argued that the evidence was inadmissible",
    "It was submitted that public interest required dismissal",
    "The appellant maintained that the duty of care was established",
    "The respondent objected to paragraph 12 of the affidavit",
    "Senior counsel submitted that the provision should be read narrowly",
    "The appellant contended in the alternative that damages were excessive",

    # EDGE CASE CLARIFICATION
    "Includes transcript passages where counsel advances a legal position",
    "Includes evidentiary and procedural arguments",
    "Includes policy based and textual interpretation arguments",
    "Statements explicitly attributed to a party are arguments",

    # NEGATIVE BOUNDARY
    "Does not contain the court's evaluation of those arguments",
    "Does not state the final determination",
    "Does not merely identify the issue without advancing a position",
    "Does not narrate factual background unless used as part of submission",
    "Judicial rejection or acceptance of a submission is not arguments"
],


"legal_reasoning": [

    # FUNCTION
    "Judicial analysis explanation and interpretation of law",
    "Application of legal principles to established facts",
    "Evaluation of competing arguments",

    # TYPICAL CONTENT EXPANDED
    "Quotation and interpretation of statutory provisions",
    "Discussion of precedent and authority",
    "Logical inferences drawn from legislative language",
    "Assessment of persuasiveness of submissions",
    "Application of legal tests to factual findings",
    "Evaluation of credibility or evidentiary sufficiency",
    "Discussion of burden of proof",
    "Consideration of policy rationale underlying statute",
    "Resolution of ambiguity in statutory or common law principles",

    # STRONG EXAMPLES
    "Section 154 must be read in its statutory context",
    "In our view the language of the Act is clear",
    "It follows that the submission cannot be sustained",
    "I am not satisfied that the evidence establishes negligence",
    "The burden of proof lies on the applicant",
    "Authority establishes that consideration is required",
    "To accept that argument would be inconsistent with precedent",
    "The statutory scheme does not leave the matter at large",
    "In my opinion the correct construction is",
    "The submission must fail because it is inconsistent with authority",

    # EDGE CASE CLARIFICATION
    "Includes statutory quotation accompanied by interpretation",
    "Includes mixed fact law analysis",
    "Includes statements of satisfaction or dissatisfaction",
    "Includes reasoning immediately preceding final orders",
    "Application of legal test to established facts is legal reasoning",

    # NEGATIVE BOUNDARY
    "Does not merely describe procedural events",
    "Does not simply restate party submissions",
    "Does not consist solely of identifying the issue",
    "Does not contain the formal operative order"
],


"decision": [

    # FUNCTION
    "Final authoritative determination resolving the dispute",
    "Formal operative pronouncement of the court or tribunal",

    # TYPICAL CONTENT EXPANDED
    "Dismissal or allowance of appeal",
    "Confirmation variation or setting aside of decision",
    "Grant or refusal of relief",
    "Cost orders including no order as to costs",
    "Declarations of rights or obligations",
    "Orders remitting matter to lower court",
    "Imposition of liability",

    # STRUCTURAL SIGNALS
    "Text beginning with THE COURT ORDERS THAT",
    "Text beginning with THE TRIBUNAL ORDERS THAT",
    "Formal numbered orders",
    "Judgment is entered for",
    "The appeal is dismissed",
    "The application is granted",

    # STRONG EXAMPLES
    "The appeal is dismissed",
    "The decision below is set aside",
    "There be no order as to costs",
    "The respondent must pay the applicant's costs",
    "The licence scheme is confirmed",
    "Judgment is entered for the defendant",
    "The matter is remitted for rehearing",
    "The application is refused",
    "Costs to follow the event",

    # NEGATIVE BOUNDARY
    "Does not merely frame the issue",
    "Does not provide extended reasoning",
    "Does not narrate factual background",
    "Does not describe procedural history",
    "Does not consist solely of headings or metadata"
],


"other": [

    # FUNCTION
    "Material without substantive legal analytical function",
    "Administrative or formatting content unrelated to legal reasoning",

    # TYPICAL CONTENT EXPANDED
    "Page numbers running headers or footers",
    "Editorial publication metadata",

    # STRONG EXAMPLES
    "Page 12",
    "Downloaded from www.example.com",
    "IN THE HIGH COURT OF AUSTRALIA",
    "No 1 of 1991",
    "Copyright notice",
    "END OF DOCUMENT"
]

}
