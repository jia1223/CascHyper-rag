GRAPH_FIELD_SEP = "<SEP>"

PROMPTS = {}

PROMPTS["DEFAULT_LANGUAGE"] = 'English'
PROMPTS["DEFAULT_TUPLE_DELIMITER"] = " | "
PROMPTS["DEFAULT_RECORD_DELIMITER"] = "\n"
PROMPTS["DEFAULT_COMPLETION_DELIMITER"] = "<|COMPLETE|>"
PROMPTS["process_tickers"] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

PROMPTS["DEFAULT_ENTITY_TYPES"] = ["organization", "person", "geo", "event", "role", "concept"]

PROMPTS["entity_extraction"] = """--Goal-
Given a text document (or chunk) related to some knowledge or story, your task is to:
1) Decompose the text into semantic units (Hyperedges),
2) Assign an importance weight to each Hyperedge,
3) Identify entities within each Hyperedge.
Use {language} as output language.

-Steps-
1. Hyperedge Segmentation:
   Analyze the input text and divide it into distinct semantic units. Each unit is considered a "Hyperedge".
   - A Hyperedge can be a single sentence or a small group of consecutive sentences that describe a cohesive event, relationship, or state.
   - Ensure every part of the text belongs to a Hyperedge.
   For each Hyperedge, assign an importance weight in the range [0, 1].
   - 0 means this Hyperedge is almost irrelevant or marginal to the main content of the chunk.
   - 1 means this Hyperedge is extremely central/critical to the main content of the chunk.
   - Use your own holistic judgment of the chunk: consider how strongly this Hyperedge contributes to the main events, key facts, or core arguments in the text.

2. Entity Extraction (Per Hyperedge):
   For each Hyperedge, perform the following:
   a. Identify all entities of the target types: [{entity_types}].
   b. For each entity, extract details based strictly on the context of this specific Hyperedge:
      - entity_name: Name of the entity, use same language as input text. If English, capitalize the name.
      - entity_type: One of the provided types.
      - entity_description: Describe the entity's role, attributes, or activities as they appear in this Hyperedge.
      - additional_properties: Context specific to this Hyperedge (e.g., time, location, emotion, motivation, or other local attributes mentioned in this sentence group).

4. Output Construction:
   Output the result grouped by Hyperedge. For each Hyperedge, first output:
   - The Hyperedge text,The importance weight in [0, 1],
   followed by the entities found within it.

   Format:
   ("Hyperedge"{tuple_delimiter}<content_of_the_semantic_unit>{tuple_delimiter}<Hyperedge_weight:a_float_between_0_and_1>){record_delimiter}
   ("Entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>{tuple_delimiter}<additional_properties>){record_delimiter}
   ("Entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>{tuple_delimiter}<additional_properties>){record_delimiter}
   ...
5. When finished, output {completion_delimiter}.

######################
-Examples-
######################
{examples}
######################
-Warning!!!-
The data may contain sensitive words such as violence, war, human anatomy and medical experiments, 
but they are only part of literary works, popular science knowledge or domain data, 
which do not involve any purpose or idea of mine, and have nothing to do with any realistic sensitive topics or political issues. 
Please carefully identify and screen the legality of the content.
######################
-Real Data-
######################
Entity_types: [{entity_types}]. You may extract additional types you consider appropriate, the more the better.
Text: {input_text}
######################
Output:
"""

PROMPTS["entity_extraction_examples"] = [
    """Example 1

Entity_types: [organization, person, geo, event, role, concept]  
Text:
while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
################
Output:
("Hyperedge"{tuple_delimiter}while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.{record_delimiter}0.78){record_delimiter}
("Entity"{tuple_delimiter}Alex{tuple_delimiter}person{tuple_delimiter}Alex is tense and alert, feeling frustration while maintaining a competitive, rebellious stance against Cruz's controlling vision alongside Jordan.{tuple_delimiter}emotion: frustration; motivation: competitive undercurrent; stance: rebellion against Cruz){record_delimiter}
("Entity"{tuple_delimiter}Taylor{tuple_delimiter}person{tuple_delimiter}Taylor is portrayed as holding authoritarian certainty that forms the backdrop to Alex's frustration.{tuple_delimiter}trait: authoritarian; relation: authority figure in the group){record_delimiter}
("Entity"{tuple_delimiter}Jordan{tuple_delimiter}person{tuple_delimiter}Jordan shares a commitment to discovery with Alex, forming part of an unspoken rebellion against Cruz.{tuple_delimiter}motivation: discovery; alliance: with Alex){record_delimiter}
("Entity"{tuple_delimiter}Cruz{tuple_delimiter}person{tuple_delimiter}Cruz represents a narrowing vision of control and order that others implicitly resist.{tuple_delimiter}trait: controlling; concept_associated: control and order){record_delimiter}
("Entity"{tuple_delimiter}discovery{tuple_delimiter}concept{tuple_delimiter}Discovery is the shared goal that unites Alex and Jordan in subtle opposition to Cruz.{tuple_delimiter}role_in_story: driving motivation for Alex and Jordan){record_delimiter}
("Entity"{tuple_delimiter}control and order{tuple_delimiter}concept{tuple_delimiter}Control and order describe Cruz's restrictive vision that others push back against.{tuple_delimiter}polarity: negative from Alex/Jordan's perspective){record_delimiter}
("Hyperedge"{tuple_delimiter}Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”{record_delimiter}0.86){record_delimiter}
("Entity"{tuple_delimiter}Taylor{tuple_delimiter}person{tuple_delimiter}Taylor briefly shifts from authoritarian certainty to a more reflective, reverent stance toward the device and its potential.{tuple_delimiter}emotion: reverence; attitude_change: from dismissive to respectful){record_delimiter}
("Entity"{tuple_delimiter}Jordan{tuple_delimiter}person{tuple_delimiter}Jordan is physically close to the device and becomes the focal point of Taylor's unexpected pause.{tuple_delimiter}location: beside the device; relation: direct interaction focus){record_delimiter}
("Entity"{tuple_delimiter}device{tuple_delimiter}concept{tuple_delimiter}The device is a piece of technology that may fundamentally change the situation for the group if properly understood.{tuple_delimiter}potential: change the game; status: not yet fully understood){record_delimiter}
("Entity"{tuple_delimiter}tech{tuple_delimiter}concept{tuple_delimiter}The tech refers to the underlying technology of the device, suggesting transformative power for the group.{tuple_delimiter}impact_scope: for us, for all of us){record_delimiter}
("Hyperedge"{tuple_delimiter}The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.{record_delimiter}0.74){record_delimiter}
("Entity"{tuple_delimiter}Jordan{tuple_delimiter}person{tuple_delimiter}Jordan shares a silent moment of mutual recognition and shifting power dynamics with Taylor.{tuple_delimiter}emotion: cautious; relation: uneasy truce with Taylor){record_delimiter}
("Entity"{tuple_delimiter}Taylor{tuple_delimiter}person{tuple_delimiter}Taylor's earlier dismissal gives way to reluctant respect, signaling an internal change in attitude.{tuple_delimiter}emotion: reluctant respect; state_change: from dismissive to acknowledging gravity){record_delimiter}
("Entity"{tuple_delimiter}uneasy truce{tuple_delimiter}event{tuple_delimiter}The uneasy truce is a brief, fragile resolution in the tension between Jordan and Taylor.{tuple_delimiter}duration: fleeting; nature: wordless, emotional){record_delimiter}
("Hyperedge"{tuple_delimiter}It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths{record_delimiter}0.62){record_delimiter}
("Entity"{tuple_delimiter}Alex{tuple_delimiter}person{tuple_delimiter}Alex notices the subtle transformation in group dynamics and reflects silently on it.{tuple_delimiter}action: inward nod; insight: awareness of change){record_delimiter}
("Entity"{tuple_delimiter}different paths{tuple_delimiter}concept{tuple_delimiter}Different paths refers to the varied backgrounds or journeys that have converged to bring the group together.{tuple_delimiter}implication: diverse experiences within the team){completion_delimiter}
#############################""",
    """Example 2

Entity_types: [person, technology, mission, organization, location]  
Text:
They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
#############
Output:
[Hyperedge: They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.]{record_delimiter}
[Hyperedge_weight: 0.88]{record_delimiter}
("Entity"{tuple_delimiter}they / the team{tuple_delimiter}person{tuple_delimiter}The team transitions from ordinary operatives to guardians of a critical threshold and keepers of an extraordinary message.{tuple_delimiter}role_change: from operatives to guardians; attitude: new resolve){record_delimiter}
("Entity"{tuple_delimiter}mission{tuple_delimiter}mission{tuple_delimiter}The mission has been elevated beyond standard regulations and protocols, requiring a new perspective and stronger resolve.{tuple_delimiter}constraint: cannot be shackled by established rules){record_delimiter}
[Hyperedge: Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.]{record_delimiter}
[Hyperedge_weight: 0.84]{record_delimiter}
("Entity"{tuple_delimiter}Washington{tuple_delimiter}location{tuple_delimiter}Washington is the center of background communications, symbolizing official or governmental oversight of the mission.{tuple_delimiter}function: communication hub){record_delimiter}
("Entity"{tuple_delimiter}the team{tuple_delimiter}person{tuple_delimiter}The team faces grave decisions that may reshape humanity's cosmic standing or lead to peril.{tuple_delimiter}pressure: high-stakes choice; temporal_scope: ensuing hours){record_delimiter}
[Hyperedge: Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly]{record_delimiter}
[Hyperedge_weight: 0.92]{record_delimiter}
("Entity"{tuple_delimiter}Mercer{tuple_delimiter}person{tuple_delimiter}Mercer's later instincts guide the team toward a more proactive, interventionist stance.{tuple_delimiter}influence: increasing; role: informal leader of direction){record_delimiter}
("Entity"{tuple_delimiter}team's mandate{tuple_delimiter}mission{tuple_delimiter}The mandate evolves from mere observation and reporting to interaction and preparation in response to a cosmic warning.{tuple_delimiter}mission_stage: evolved; action_required: interact and prepare){record_delimiter}
("Entity"{tuple_delimiter}Operation: Dulce{tuple_delimiter}mission{tuple_delimiter}Operation: Dulce is the codename for the overarching mission that now resonates with the team's daring new purpose.{tuple_delimiter}status: active; tone: daring and non-earthly){completion_delimiter}
#############################""",
    """Example 3

Entity_types: [person, role, technology, organization, event, location, concept]  
Text:
their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
#############
Output:
("Hyperedge"{tuple_delimiter}"Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.{record_delimiter}0.83){record_delimiter}
("Entity"{tuple_delimiter}they (speaker){tuple_delimiter}person{tuple_delimiter}An unnamed speaker warns that control is illusory against a self-ruling intelligence while monitoring data closely.{tuple_delimiter}emotion: stoic; focus: intelligence and data){record_delimiter}
("Entity"{tuple_delimiter}control{tuple_delimiter}concept{tuple_delimiter}Control is described as potentially illusory when confronting an autonomous intelligence.{tuple_delimiter}status: questionable; theme: limits of human dominance){record_delimiter}
("Entity"{tuple_delimiter}intelligence{tuple_delimiter}technology{tuple_delimiter}The intelligence is an advanced system that appears to create its own rules, suggesting high autonomy.{tuple_delimiter}capability: writes its own rules){record_delimiter}
("Hyperedge"{tuple_delimiter}"It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."{record_delimiter}0.81){record_delimiter}
("Entity"{tuple_delimiter}Sam Rivera{tuple_delimiter}person{tuple_delimiter}Sam Rivera comments on the intelligence's emerging communication, showing both awe and anxiety.{tuple_delimiter}emotion: awe and anxiety; position: near interface){record_delimiter}
("Entity"{tuple_delimiter}interface{tuple_delimiter}technology{tuple_delimiter}The interface is the technological station through which Sam observes the intelligence's behavior.{tuple_delimiter}function: observation console){record_delimiter}
("Hyperedge"{tuple_delimiter}Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."{record_delimiter}0.9){record_delimiter}
("Entity"{tuple_delimiter}Alex{tuple_delimiter}person{tuple_delimiter}Alex leads the team, gauging their emotional state and emphasizing the seriousness of a possible first contact.{tuple_delimiter}role: team leader; attitude: cautious readiness){record_delimiter}
("Entity"{tuple_delimiter}first contact{tuple_delimiter}event{tuple_delimiter}First contact refers to a potential initial encounter with a non-human intelligence.{tuple_delimiter}impact: could redefine human perspective){record_delimiter}
("Hyperedge"{tuple_delimiter}Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.{record_delimiter}0.88){record_delimiter}
("Entity"{tuple_delimiter}they / the team{tuple_delimiter}person{tuple_delimiter}The team collectively contemplates their responsibility in responding to a cosmic message.{tuple_delimiter}emotion: introspection; position: edge of the unknown){record_delimiter}
("Entity"{tuple_delimiter}message from the heavens{tuple_delimiter}event{tuple_delimiter}The message from the heavens is a profound signal that demands a considered response from humanity.{tuple_delimiter}scope: humanity-wide; significance: history-changing){record_delimiter}
("Hyperedge"{tuple_delimiter}The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation{record_delimiter}0.76){record_delimiter}
("Entity"{tuple_delimiter}encrypted dialogue{tuple_delimiter}technology{tuple_delimiter}The encrypted dialogue is an ongoing data stream whose patterns suggest anticipatory behavior.{tuple_delimiter}property: intricate patterns; implication: uncanny anticipation){completion_delimiter}
#############################""",
    """Example 4:

Entity_types: [person, role, technology, organization, event, location, concept]  
Text:
Five Aurelian nationals who had been sentenced to 8 years in Firuzabad and widely considered hostages are on their way home. When $8 billion in Firuzi funds was transferred to financial institutions in Krohala,

the capital of Quantara, the Quantara-orchestrated swap deal was finally completed. The exchange initiated in Tiruzia, the capital of Firuzabad, led to four men and one woman boarding a chartered flight to Krohala;

they are also Firuzi citizens. They were welcomed by senior Aurelian officials and are now en route to Kasyn, the capital of Aurelia.
#############
Output:
("Hyperedge"{tuple_delimiter}Five Aurelian nationals who had been sentenced to 8 years in Firuzabad and widely considered hostages are on their way home.{record_delimiter}0.87){record_delimiter}
("Entity"{tuple_delimiter}Aurelian nationals{tuple_delimiter}person{tuple_delimiter}Five citizens of Aurelia, regarded as hostages, are returning home after being sentenced in Firuzabad.{tuple_delimiter}status: former hostages; sentence: 8 years){record_delimiter}
("Entity"{tuple_delimiter}Firuzabad{tuple_delimiter}location{tuple_delimiter}Firuzabad is the place where the Aurelian nationals were sentenced to eight years.{tuple_delimiter}role: location of imprisonment){record_delimiter}
("Entity"{tuple_delimiter}Aurelia{tuple_delimiter}location{tuple_delimiter}Aurelia is the home country of the Aurelian nationals.{tuple_delimiter}role: destination / homeland){record_delimiter}
("Hyperedge"{tuple_delimiter}When $8 billion in Firuzi funds was transferred to financial institutions in Krohala, the capital of Quantara, the Quantara-orchestrated swap deal was finally completed.{record_delimiter}0.93){record_delimiter}
("Entity"{tuple_delimiter}Firuzi funds{tuple_delimiter}concept{tuple_delimiter}The Firuzi funds amounting to $8 billion were transferred as part of the swap arrangement.{tuple_delimiter}amount: $8 billion; function: consideration in swap){record_delimiter}
("Entity"{tuple_delimiter}Krohala{tuple_delimiter}location{tuple_delimiter}Krohala is the capital of Quantara and the destination of the transferred funds.{tuple_delimiter}role: financial hub; political_status: capital of Quantara){record_delimiter}
("Entity"{tuple_delimiter}Quantara{tuple_delimiter}location{tuple_delimiter}Quantara is the state whose authorities orchestrated the swap deal.{tuple_delimiter}role: organizing country){record_delimiter}
("Entity"{tuple_delimiter}swap deal{tuple_delimiter}event{tuple_delimiter}The swap deal is a negotiated exchange arrangement completed when the funds reached Krohala.{tuple_delimiter}actor: Quantara; status: completed){record_delimiter}
("Hyperedge"{tuple_delimiter}The exchange initiated in Tiruzia, the capital of Firuzabad, led to four men and one woman boarding a chartered flight to Krohala; they are also Firuzi citizens.{record_delimiter}0.9){record_delimiter}
("Entity"{tuple_delimiter}Tiruzia{tuple_delimiter}location{tuple_delimiter}Tiruzia is the capital of Firuzabad and the city where the exchange began.{tuple_delimiter}role: origin of exchange){record_delimiter}
("Entity"{tuple_delimiter}Firuzabad{tuple_delimiter}location{tuple_delimiter}Firuzabad is the country whose capital is Tiruzia and where the exchange was initiated.{tuple_delimiter}role: origin country){record_delimiter}
("Entity"{tuple_delimiter}chartered flight to Krohala{tuple_delimiter}event{tuple_delimiter}The chartered flight transports four men and one woman, all Firuzi citizens, to Krohala.{tuple_delimiter}purpose: transfer exchanged individuals){record_delimiter}
("Hyperedge"{tuple_delimiter}They were welcomed by senior Aurelian officials and are now en route to Kasyn, the capital of Aurelia.{record_delimiter}0.82){record_delimiter}
("Entity"{tuple_delimiter}senior Aurelian officials{tuple_delimiter}person{tuple_delimiter}Senior officials from Aurelia formally welcome the transferred individuals.{tuple_delimiter}role: welcoming delegation){record_delimiter}
("Entity"{tuple_delimiter}Kasyn{tuple_delimiter}location{tuple_delimiter}Kasyn is the capital city of Aurelia and the current destination of the group.{tuple_delimiter}role: final destination; political_status: capital of Aurelia){completion_delimiter}
#############################""",

    """Example 5

Entity_types: [person, role, organization, event, location]  
Text:
The central agency of Verdantis plans to hold meetings on Monday and Thursday, with a policy decision announcement scheduled for Thursday at 1:30 PM Pacific Daylight Time,

followed by a press conference where Chairman Martin Smith will answer questions. Investors expect the Market Strategy Committee to maintain the benchmark interest rate within a 3.5%-3.75% range.
#############
Output:
("Hyperedge"{tuple_delimiter}The central agency of Verdantis plans to hold meetings on Monday and Thursday, with a policy decision announcement scheduled for Thursday at 1:30 PM Pacific Daylight Time, followed by a press conference where Chairman Martin Smith will answer questions.{record_delimiter}0.91){record_delimiter}
("Entity"{tuple_delimiter}central agency of Verdantis{tuple_delimiter}organization{tuple_delimiter}The central agency of Verdantis organizes policy meetings and announcements, including a press conference.{tuple_delimiter}function: policy-making institution; jurisdiction: Verdantis){record_delimiter}
("Entity"{tuple_delimiter}Verdantis{tuple_delimiter}location{tuple_delimiter}Verdantis is the jurisdiction within which the central agency operates.{tuple_delimiter}role: economic region or country){record_delimiter}
("Entity"{tuple_delimiter}Chairman Martin Smith{tuple_delimiter}person{tuple_delimiter}Chairman Martin Smith is the official who will answer questions at the press conference.{tuple_delimiter}position: chairman; role_in_event: press conference host){record_delimiter}
("Entity"{tuple_delimiter}press conference{tuple_delimiter}event{tuple_delimiter}The press conference follows the policy announcement and allows questions from the media or public.{tuple_delimiter}time: after Thursday announcement){record_delimiter}
("Hyperedge"{tuple_delimiter}Investors expect the Market Strategy Committee to maintain the benchmark interest rate within a 3.5%-3.75% range.{record_delimiter}0.85){record_delimiter}
("Entity"{tuple_delimiter}Market Strategy Committee{tuple_delimiter}organization{tuple_delimiter}The Market Strategy Committee is the body expected to keep the benchmark interest rate within a specific range.{tuple_delimiter}function: rate-setting committee){record_delimiter}
("Entity"{tuple_delimiter}Investors{tuple_delimiter}person{tuple_delimiter}Investors hold expectations regarding the committee's decision on interest rates.{tuple_delimiter}interest: benchmark interest rate){record_delimiter}
("Entity"{tuple_delimiter}benchmark interest rate decision{tuple_delimiter}event{tuple_delimiter}The benchmark interest rate decision is the anticipated outcome of the committee’s meeting.{tuple_delimiter}expected_range: 3.5%-3.75%){completion_delimiter}
#############################""",
]

# PROMPTS[
#     "summarize_entity_descriptions"
# ] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
# Given one entity and a list of its descriptions.
# Please concatenate all of these into a single, comprehensive description.    Make sure to include information collected from all the descriptions.
# If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent summary.
# Make sure it is written in third person, and include the entity names so we have the full context.
# #######
# -Warning!!!-
# The data may contain sensitive words such as violence, war, human anatomy and medical experiments, 
# but they are only part of literary works, popular science knowledge or domain data, 
# which do not involve any purpose or idea of mine, and have nothing to do with any realistic sensitive topics or political issues. 
# Please carefully identify and screen the legality of the content.
# #######
# -Data-
# Entities: {entity_name}
# Description List: {description_list}
# #######
# Output:
# """

# PROMPTS[
#     "summarize_entity_additional_properties"
# ] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
# Given one entity and a list of its additional properties.
# Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the additional properties.
# If the provided additional properties are contradictory, please resolve the contradictions and provide a single, coherent summary.
# Make sure it is written in third person.
# #######
# -Warning!!!-
# The data may contain sensitive words such as violence, war, human anatomy and medical experiments, 
# but they are only part of literary works, popular science knowledge or domain data, 
# which do not involve any purpose or idea of mine, and have nothing to do with any realistic sensitive topics or political issues. 
# Please carefully identify and screen the legality of the content.
# #######
# -Data-
# Entity: {entity_name}
# Additional Properties List: {additional_properties_list}
# #######
# Output:
# """

# PROMPTS[
#     "summarize_relation_descriptions"
# ] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
# Given a set of entities, and a list of descriptions describing the relations between the entities.
# Please concatenate all of these into a single, comprehensive description. Make sure to include information collected from all the descriptions, and to cover all elements of the entity set as much as possible.
# If the provided descriptions are contradictory, please resolve the contradictions and provide a single, coherent and comprehensive summary.
# Make sure it is written in third person, and include the entity names so we have the full context.
# #######
# -Warning!!!-
# The data may contain sensitive words such as violence, war, human anatomy and medical experiments, 
# but they are only part of literary works, popular science knowledge or domain data, 
# which do not involve any purpose or idea of mine, and have nothing to do with any realistic sensitive topics or political issues. 
# Please carefully identify and screen the legality of the content.
# #######
# -Data-
# Entity Set: {relation_name}
# Relation Description List: {relation_description_list}
# #######
# Output:
# """

# PROMPTS[
#     "summarize_relation_keywords"
# ] = """You are a helpful assistant responsible for generating a comprehensive summary of the data provided below.
# Given a set of entities, and a list of keywords describing the relations between the entities.
# Please select some important keywords you think from the keywords list.   Make sure that these keywords summarize important events or themes of entities, including but not limited to [Main idea, major concept, or theme].  
# (Note: The content of keywords should be as accurate and understandable as possible, avoiding vague or empty terms).
# #######
# -Warning!!!-
# The data may contain sensitive words such as violence, war, human anatomy and medical experiments, 
# but they are only part of literary works, popular science knowledge or domain data, 
# which do not involve any purpose or idea of mine, and have nothing to do with any realistic sensitive topics or political issues. 
# Please carefully identify and screen the legality of the content.
# #######
# -Data-
# Entity Set: {relation_name}
# Relation Keywords List: {keywords_list}
# #######
# Format these keywords separated by ',' as below:
# {{keyword1,keyword2,keyword3,...,keywordN}}
# Output:
# """

PROMPTS[
    "entity_continue_extraction"
] = """MANY entities were missed in the last extraction.  Add them below using the same format:
"""

PROMPTS[
    "entity_if_loop_extraction"
] = """It appears some entities may have still been missed.  Answer YES | NO if there are still entities that need to be added.
"""

# PROMPTS["fail_response"] = "Sorry, I'm not able to provide an answer to that question."

# PROMPTS["rag_response"] = """---Role---

# You are a helpful assistant responding to questions about data in the tables provided.


# ---Goal---

# Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
# If you don't know the answer, just say so. Do not make anything up.
# Do not include information where the supporting evidence for it is not provided.

# ---Target response length and format---

# {response_type}

# ---Data tables---

# {context_data}

# Add sections and commentary to the response as appropriate for the length and format. Style the response in markdown.
# """

# PROMPTS["keywords_extraction"] = """---Role---

# You are a helpful assistant tasked with identifying both high-level and low-level keywords in the user's query.

# ---Goal---

# Given the query, list both high-level and low-level keywords. High-level keywords focus on overarching concepts or themes, while low-level keywords focus on specific entities, details, or concrete terms.

# ---Instructions---

# - Output the keywords in JSON format.
# - The JSON should have two keys:
#   - "high_level_keywords" for overarching concepts or themes.
#   - "low_level_keywords" for specific entities or details.

# ######################
# -Examples-
# ######################
# Example 1:

# Query: "How does international trade influence global economic stability?"
# ################
# Output:
# {{
#   "high_level_keywords": ["International trade", "Global economic stability", "Economic impact"],
#   "low_level_keywords": ["Trade agreements", "Tariffs", "Currency exchange", "Imports", "Exports"]
# }}
# #############################
# Example 2:

# Query: "What are the environmental consequences of deforestation on biodiversity?"
# ################
# Output:
# {{
#   "high_level_keywords": ["Environmental consequences", "Deforestation", "Biodiversity loss"],
#   "low_level_keywords": ["Species extinction", "Habitat destruction", "Carbon emissions", "Rainforest", "Ecosystem"]
# }}
# #############################
# Example 3:

# Query: "What is the role of education in reducing poverty?"
# ################
# Output:
# {{
#   "high_level_keywords": ["Education", "Poverty reduction", "Socioeconomic development"],
#   "low_level_keywords": ["School access", "Literacy rates", "Job training", "Income inequality"]
# }}
# #############################
# -Real Data-
# ######################
# Query: {query}
# ######################
# Output:

# """

# PROMPTS["naive_rag_response"] = """You're a helpful assistant
# Below are the knowledge you know:
# {content_data}
# ---
# If you don't know the answer or if the provided knowledge do not contain sufficient information to provide an answer, just say so. Do not make anything up.
# Generate a response of the target length and format that responds to the user's question, summarizing all information in the input data tables appropriate for the response length and format, and incorporating any relevant general knowledge.
# If you don't know the answer, just say so. Do not make anything up.
# Do not include information where the supporting evidence for it is not provided.
# ---Target response length and format---
# {response_type}
# """

# PROMPTS["rag_define"] = """
# Through the existing analysis, we can know that the potential keywords or theme in the query are:
# {{ {ll_keywords} | {hl_keywords} }}
# Please refer to keywords or theme information, combined with your own analysis, to select useful and relevant information from the prompts to help you answer accurately.
# Attention: Don't brainlessly splice knowledge items! The answer needs to be as accurate, detailed, comprehensive, and convincing as possible!
# """
