from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel

prompt_1 = PromptTemplate(
    template='Generate short and simple notes from the following text: \n {text}',
    input_variables=['text']
)

prompt_2 = PromptTemplate(
    template='Generate generate 5 short qna from the following text: \n {text}',
    input_variables=['text']
)

prompt_3 = PromptTemplate(
    template='Merge the provided notes and qna into a single document: \n {notes} and {qna}',
    input_variables=['notes', 'qna']
)

model_1 = ChatOllama(model='llama2')
model_2 = ChatOllama(model='deepseek-r1')

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes': prompt_1 | model_1 | parser,
    'qna': prompt_2 | model_2 | parser
})

merge_chain = prompt_3 | model_2 | parser

chain = parallel_chain | merge_chain

text = '''The ideas behind Expedition 33 originated in 2019 with Guillaume Broche, an employee of Ubisoft, not long before the COVID-19 pandemic;[35] it soon grew into a passion project inspired by his childhood favorites, most notably the Final Fantasy series.[36] He sent out some requests for help to craft a demo to a group of other developers he knew, along with requests on Reddit in April 2020 looking for voice actors for said demo. In order to focus on his project full-time, Broche would leave Ubisoft that year and form Sandfall Interactive, alongside François Meurisse, an old friend, and fellow Ubisoft developer Tom Guillermin. The three co-founders would soon be joined by Lorien Testard, Nicholas Maxson-Francombe and Jennifer Svedberg-Yen; the six forming the kick-off team.[36][35] Lorien Testard, the composer, was discovered by Broche through a post on a French indie video game forum where he linked a track from his SoundCloud page.[37][36][38] Maxson-Francombe, the game's art director, was discovered and recruited off ArtStation by Broche. Svedberg-Yen, one of the voice actors who had stumbled upon Broche's Reddit post and was cast for the original demo, gained a more prominent role as development progressed, becoming the game's lead writer.[36]

After inking a partnership with Kepler Interactive, which was officially announced in early 2023, and securing funding from said publisher, Sandfall grew into a studio of about thirty developers, three of whom—including Broche and Guillermin—were former Ubisoft developers.[39][40][41][30][28][31][excessive citations] The funding also allowed Sandfall to expand the manpower contributing to the project beyond this core team, having outsourced gameplay combat animation to a team of eight South Korean freelance animators and quality assurance (QA) to a few dozen QA testers from the firm QLOC, as well as receiving porting assistance from a half-dozen developers from Ebb Software. The studio also hired a couple of performance capture artists; brought in musicians for the soundtrack recording sessions; contracted with translators from Riotloc for language localization; and partnered with Side UK and Studio Anatole as to voice casting and production in English and French respectively.[40][42] Finally, the partnership with Kepler Interactive enabled Sandfall to pay for noted professional voice actors, including Charlie Cox, Andy Serkis and Ben Starr.[36][38] Cox has jokingly stated he was in the studio for barely four hours to record his lines and felt like a fraud over being lauded for his performance; though Svedberg-Yen clarified on her Instagram account that the task had taken around 8 hours, and praised his professionalism and efficiency.[43] The total budget was less than $10 million.[44]

Broche stated that the purpose of Expedition 33 was to create a high fidelity turn-based RPG, which he felt had been neglected by AAA game developers.[27] Besides Final Fantasy, Expedition 33 took inspiration from other Japanese role-playing games, including the Persona series;[45] Broche notably praised Persona 5 for its user interface and use of camera work during battles, "making it feel like you're watching a movie".[46] Broche also considered Lost Odyssey and Blue Dragon, JRPGs developed for Microsoft's drive to help market the Xbox consoles in Japan, as an influence, particularly their use of quick time events during combat.[47] According to producer François Meurisse, the game drew inspiration from SquareSoft's Final Fantasy VIII, Final Fantasy IX and Final Fantasy X in particular, while the dodge and parry mechanics were influenced by FromSoftware's Sekiro: Shadows Die Twice.[48]


Screenshot of a demo trailer, released via a Reddit post by Broche in 2020, for We Lost, which would later become Expedition 33. The game's setting originally took inspiration from the steampunk subgenre.
Development initially began under the codename "Project W",[49] and was first known as We Lost around the time that Broche sought help on Reddit. The initial demo showcased a steampunk setting inspired by Victorian era England, with more science fiction elements, including zombies and aliens. About six months into this approach, potential investors suggested that Broche should "think bigger" and ponder what he would want to do if they weren't restricted by their limited resources. This led him to reset the entire story, opting for the Belle Époque—a period the French team was naturally well acquainted with and which they deemed to be a more distinctive setting—as well as taking inspiration from the Art Deco movement often associated with the era for the visual world design.[50] The new narrative was based on a painting Broche admired, which led him to think of a giantess and a doomsday clock, while also taking some inspiration from the French fantasy novel La Horde du Contrevent by Alain Damasio. The latter featured a horde of men, trained since childhood, undertaking an odyssey to reach the mythical "Extrême-Amont", the source of all winds.[51] Broche's premise was then associated with a short story unrelated to the project Svedberg-Yen had written on her own for fun, in which a painter capable of traveling through her own works got lost in one, prompting her daughter's endeavor to save her.[52][53]

Svedberg-Yen stated one of the game's core themes, the loss of loved ones, originates with Broche's mother and represented the "final piece". As the two were stuck on the draft, Broche asked his mother what would be the worst thing that could happen to her; she answered the loss of any of her children. This notably became the foundation for Aline's character and became the catalyst for her decision to dwell in what Svedberg-Yen and Broche subsequently conceived as her departed son's canvas.[52][54] While other aspects of the narrative were crafted as the game progressed, Svedberg-Yen asserted that the ending of Act I, featuring the death of Gustave, was something she and Broche had set early on, as part of the emotional journey they wanted for the characters.[54] The team kept some of the characters they had already envisioned in We Lost, such as Maelle and Lune, but their design and characterization were reassessed accordingly. The characters Noco and Monoco derive their names from the Swedish energy drink Nocco. Broche and Guillermin met during their stint at a subsidiary of Ubisoft in Malmö, Sweden, where a refrigerator was reportedly filled with the beverage. The studio's dog, Monoco, who's credited as Sandfall's "Happiness Manager" on the studio's website, is named after the character;[52] meanwhile, Svedberg-Yen's own dog, Trunks, was an inspiration for the latter's character design, notably his mop-like hair style.[55]

Development began with Unreal Engine 4, switching to Unreal Engine 5 due to its improvements in rendering and animation.[28][31] The engine's Nanite and Lumen features allowed for higher-fidelity assets and more-realistic lighting, respectively.[56] The adoption of Lumen necessitated reworking the lighting for most environments.[56] Additionally, UE5 had more support for character creation—an advantage over Reallusion's Character Creator, previously being used in this regard.[56] Sandfall relied on ready-made assets for background objects such as rocks, enabling them to focus on creating "hero assets", i.e., large-scale assets that make an impression on the viewer.[56] Broche credits the development of the game to the simplicity of modern engines.[36]
'''

res = chain.invoke({'text': text})

print(res)

chain.get_graph().print_ascii()