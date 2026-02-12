ROLE_DESCRIPTIONS_DICT = {
"case_metadata": [

    # FUNCTION
    "Administrative text identifying the case rather than discussing its substance",
    "Structural header material that labels the proceeding",
    "Identification information appearing before or outside the reasoning section",

    # TYPICAL CONTENT EXPANDED
    "Name of the court tribunal or judicial body",
    "Case title listing parties such as Applicant Respondent Appellant Plaintiff Defendant",
    "Neutral citation including year report number and jurisdiction",
    "Coram listing judges panel members or tribunal members",
    "Date of judgment hearing or delivery",
    "Proceeding number registry file number docket number",
    "Place where the matter was heard",
    "Headings such as REASONS FOR DECISION MINUTES OF ORDER CATCHWORDS",
    "Uppercase formatted case identifiers",
    "Reference brought under statutory provision lines",

    # STRONG EXAMPLES
    "IN THE HIGH COURT OF AUSTRALIA",
    "IN THE COPYRIGHT TRIBUNAL",
    "Between: John Smith Applicant and Mary Brown Respondent",
    "Coram: Chief Justice and Justices A B and C",
    "Judgment delivered on 12 October 2010",
    "Case No 1234 of 2009",
    "Registry: Sydney",
    "REFERENCE BROUGHT UNDER SECTION 154",

    # NEGATIVE BOUNDARY EXPANDED
    "Does not narrate events or describe conduct of parties",
    "Does not describe procedural steps taken during litigation",
    "Does not analyse statutes case law or legal principles",
    "Does not frame legal issues for determination",
    "Does not declare whether a claim succeeds or fails",
    "Does not contain evaluative reasoning or argumentative language"
],


"procedural_history": [

    # FUNCTION
    "Narrative of procedural developments within the litigation process",
    "Chronological description of steps taken in court or tribunal",
    "Account of how the matter progressed procedurally",

    # TYPICAL CONTENT EXPANDED
    "Filing lodging or commencement of proceedings",
    "Applications motions notices and summonses",
    "Appeals being filed granted refused or withdrawn",
    "Hearings listings adjournments and directions",
    "Service of documents and entry of appearances",
    "Submissions being filed or heard",
    "Interlocutory rulings procedural orders",
    "Requests for leave extensions or amendments",
    "Relief sought by a party before decision",
    "References being brought under statutory provisions",

    # STRONG EXAMPLES
    "The application was filed on 4 March 1992",
    "The matter was listed for directions",
    "Leave to appeal was granted",
    "The case was adjourned",
    "Notice of appeal was lodged",
    "Submissions were heard over two days",
    "The applicant sought the following declarations",
    "Directions were made for filing affidavits",

    # NEGATIVE BOUNDARY EXPANDED
    "Does not describe real world events outside the litigation",
    "Does not interpret statutory language in detail",
    "Does not weigh evidence or assess credibility",
    "Does not state the final outcome of the case",
    "Does not merely provide header identification information"
],


"factual_background": [

    # FUNCTION
    "Description of events that occurred outside the court process",
    "Narrative of conduct transactions or relationships giving rise to dispute",
    "Contextual history explaining the controversy",

    # TYPICAL CONTENT EXPANDED
    "Contracts employment relationships or commercial arrangements",
    "Accidents incidents meetings or communications",
    "Financial transactions transfers or payments",
    "Operation of a business industry or scheme",
    "Evidence summaries from affidavits or witness testimony",
    "Historical development of a licence scheme or policy",
    "Description of industry practices or market conditions",
    "Background about membership structure or corporate organization",

    # STRONG EXAMPLES
    "The parties entered into a contract in 1989",
    "The plaintiff delivered the goods but payment was not made",
    "An accident occurred at the intersection",
    "The employee was dismissed without notice",
    "Members of the applicant constituted 8500 composers",
    "The licence fee was calculated by reference to venue capacity",
    "Music was played using recorded tracks",

    # NEGATIVE BOUNDARY EXPANDED
    "Does not describe filings hearings or court listings",
    "Does not formulate the legal question to be decided",
    "Does not interpret statutory provisions or legal doctrine",
    "Does not announce the court's final order",
    "Does not consist solely of administrative headings"
],


"legal_issues": [

    # FUNCTION
    "Explicit identification of the legal question requiring resolution",
    "Statement of the issue for determination",
    "Formulation of the precise legal dispute",

    # TYPICAL CONTENT EXPANDED
    "Text beginning with the issue is whether",
    "Text beginning with the question is whether",
    "Identification of competing interpretations of statute",
    "Statement of a contested legal standard",
    "Framing of liability or enforceability questions",
    "Clarification of scope of statutory provision",
    "Definition of the point to be determined",

    # STRONG EXAMPLES
    "The issue is whether the contract is enforceable",
    "The question is whether a duty of care arose",
    "It must be decided whether the agreement is void",
    "The central question concerns liability for negligence",
    "The dispute concerns whether damages are recoverable",

    # NEGATIVE BOUNDARY EXPANDED
    "Does not provide extended reasoning or doctrinal discussion",
    "Does not merely recount background events",
    "Does not describe procedural chronology",
    "Does not contain the final decision or order",
    "Does not evaluate evidence in detail"
],


"legal_analysis": [

    # FUNCTION
    "Reasoned explanation and interpretation of law",
    "Application of legal principles to facts",
    "Evaluation of arguments and statutory meaning",

    # TYPICAL CONTENT EXPANDED
    "Quotation or paraphrase of statutory provisions",
    "Interpretation of legislative language",
    "Discussion of precedent and authority",
    "Comparison with earlier cases",
    "Assessment of whether arguments are persuasive",
    "Explanation of legal tests or doctrinal standards",
    "Discussion of burden of proof or evidentiary threshold",
    "Consideration of policy rationale",
    "Resolution of interpretive ambiguity",

    # STRONG EXAMPLES
    "Section 52 provides that a person shall not engage in misleading conduct",
    "At common law a duty of care arises where harm is foreseeable",
    "It is well established that consideration is required",
    "The burden of proof lies on the applicant",
    "In our view the statutory language is clear",
    "I am not satisfied that the evidence establishes negligence",

    # NEGATIVE BOUNDARY EXPANDED
    "Does not simply list procedural steps",
    "Does not merely describe real world events",
    "Does not consist solely of identifying the issue",
    "Does not simply declare the final order without reasoning",
    "Does not consist of header or citation information"
],


"holdings_and_conclusions": [

    # FUNCTION
    "Final determination resolving the legal issues",
    "Formal pronouncement of the court or tribunal",
    "Authoritative statement of the outcome",

    # TYPICAL CONTENT EXPANDED
    "Dismissal allowance or granting of appeal",
    "Confirmation variation or setting aside of decision",
    "Imposition of liability or declaration of rights",
    "Cost orders including no order as to costs",
    "Mandatory language imposing obligations",
    "Statements that a party must pay or comply",
    "Final operative paragraph of judgment",
    "Orders confirming a licence scheme",

    # STRUCTURAL SIGNALS
    "Text beginning with THE COURT ORDERS THAT",
    "Text beginning with THE TRIBUNAL ORDERS THAT",
    "Formal numbered orders",

    # STRONG EXAMPLES
    "The appeal is dismissed",
    "The application is granted",
    "The respondent must pay the applicant's costs",
    "The decision below is set aside",
    "There be no order as to costs",
    "The tribunal confirms the scheme",
    "Judgment is entered for the defendant",

    # NEGATIVE BOUNDARY EXPANDED
    "Does not merely frame the issue",
    "Does not provide extended doctrinal reasoning",
    "Does not narrate factual background",
    "Does not describe earlier procedural steps",
    "Does not consist of header information or citation details"
],


"other": [

    # FUNCTION
    "Material without substantive legal argumentative function",
    "Text unrelated to facts procedure issues analysis or decision",

    # TYPICAL CONTENT EXPANDED
    "Page numbers running headers footers",
    "Editorial or publication metadata",
    "Website download notices",
    "Table of contents index navigation elements",
    "Formatting artifacts repeated headers",
    "HTML remnants or encoding artifacts",

    # STRONG EXAMPLES
    "Page 12",
    "Downloaded from www.example.com",
    "End of document",
    "Table of contents",
    "Index",
    "Copyright notice",

    # NEGATIVE BOUNDARY EXPANDED
    "Does not contain factual narrative",
    "Does not describe procedural steps",
    "Does not interpret law",
    "Does not state legal issues",
    "Does not declare judicial outcomes"
]

}
