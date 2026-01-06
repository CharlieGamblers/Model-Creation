from pathlib import Path
import sys
import pandas as pd
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Predict import predict_multiple
from Predict.config import DEFAULT_DATE

# ---- CONFIG ----
# Optional: provide the EOS roster workbook and use:
# - Riders from sheet '2025 Test'
# - Bulls from sheet 'bulls'
# The loader will look for columns named 'name'/'rider'/'rider_name' for riders, and
# 'name'/'bull'/'bull_name' for bulls. If not found on the riders sheet, it will fall back
# to columns containing 'upper' or 'lower' (preferring 'upper').
INPUT_NAME_FILE = Path(r"C:\Users\CharlieCampbell\OneDrive - Austin Gamblers\List of FA's.xlsx")

# List of riders to predict for (used if INPUT_NAME_FILE is missing)
RIDER_NAMES = [
    "Aaron Williams",
    "Afonso Quintino",
    "Alecio Ferreira da Costa",
    "Alex Junior da Silva",
    "Alex Marcilio",
    "Alison Henrique de Lima",
    "Anderson de Oliveira",
    "Andrew Stutzman",
    "Andy Bohl",
    "Anthony Hopen",
    "Ashton Sahli",
    "Austin Perkins",
    "Boudreaux Campbell",
    "Braden Richardson",
    "Brandon Chambers",
    "Briggs Madsen",
    "Brock Radford",
    "Bruno Jose Barros",
    "Bruno Roberto",
    "Brylen Dees",
    "Caden Bunch",
    "Caleb Cantrell",
    "Carlos Andre de Oliveira",
    "Carlos Garcia",
    "Casey Coulter",
    "Casey Roberts",
    "Chad Hartman",
    "Chance Schott",
    "Chase Dougherty",
    "Chase Outlaw",
    "Chase Wimer",
    "Christian Clay Oliveira",
    "Cimarron Rucker",
    "Cladson Rodolfo",
    "Cleber Henrique Marques",
    "Cody Casper",
    "Cody Hooks",
    "Cody Teel",
    "Colby Gravier",
    "Cole Trivette",
    "Colt Robinson",
    "Colten Fritzlan",
    "Colton Byram",
    "Colton Dougherty",
    "Colton Schneiderman",
    "Conner Halverson",
    "Dakota Louis",
    "Dakota Warnken",
    "Dalton Rudman",
    "Davi Henrique de Lima",
    "David Otey",
    "Dawson Gleaves",
    "Deklan Garland",
    "Devin Hutchinson",
    "Douglas Franco",
    "Douglas Lino de Souza",
    "Dustin Herman",
    "Dustin Martinez",
    "Dustin Ratchford",
    "Dylan Smith",
    "Edgardo Figueroa",
    "Ednelio Rodrigues",
    "Eikson Pereira",
    "Eli Necochea",
    "Eli Vastbinder",
    "Emerson Lopes",
    "Ernie Courson",
    "Ethan Skogquist",
    "Ethan Weiser",
    "Felipe Soares Souza",
    "Flavio Zivieri",
    "Frederico Araujo Margarido",
    "Gabriel Morais",
    "Garfield Wilson",
    "Garrion Hull",
    "Gavan Hauck",
    "Gavin Knutson",
    "Grayson Cole",
    "Hayden Harris",
    "Jadon Hayes",
    "Jake Gardner",
    "Jake Morinec",
    "Jake Stark",
    "Jason Landing",
    "Jaxton Mortensen",
    "JC Mortensen",
    "Jean Carlos Teodoro",
    "Jean Fernandes Pereira",
    "Jerrison Begay",
    "Jerson Arantes dos Santos",
    "Jett Harkins",
    "Jhonny Klesse",
    "Joao Gabriel Saran",
    "Joao Pedro Veieria de Castro",
    "Jose Vitor Leme",
    "Josh Norman",
    "Josh Stepp",
    "JT Moore",
    "Junio Quaresima",
    "Junior Patrik Souza",
    "Kane Taylor",
    "Ky Bothum",
    "Kyler Oliver",
    "Pokey Houghton",
    "Landen Ruybal",
    "Lane Lasley",
    "Lane Nobles",
    "Lauro Nunes Vieira",
    "Leandro Machado",
    "Leonardo Lima",
    "Lorenzo Lopez",
    "Lucas Fideles Souza",
    "Lucas Martins Costa",
    "Luiz Americo P. Silva",
    "Luke Parkinson",
    "Manoelito de Souza Junior",
    "Marcelo De Souza Dias Junior",
    "Marcelo Procopio Pereira",
    "Marco Eguchi",
    "Marcos Gomes",
    "Marcus Mast",
    "Mason Moody",
    "Mason Taylor",
    "Mason Ward",
    "Matt Allgood",
    "Max Castro",
    "Merrell Cly",
    "Michael Lane",
    "Michael Phillips",
    "Natan Santos",
    "Nicolas Hernandez",
    "Paulo Henrique da Silva",
    "Perry Schrock",
    "Rafael dos Santos",
    "Rafael Parra",
    "Ramon de Lima",
    "Randy Whitener",
    "Ray Mayo",
    "Ricky Aguiar",
    "Robbie Taylor Jr.",
    "Romario Leite",
    "Rosendo Ramirez",
    "Savion Strain",
    "Scottie Knapp",
    "Shane Scott",
    "Shawn Best II",
    "Steve Martin",
    "Tevin Weston",
    "Thomas Hudson",
    "Thor Hoefer",
    "Trace Brown",
    "Trace Redd",
    "Travis Briscoe",
    "Trevor Reiste",
    "Trey Benton",
    "Tristin Parker",
    "Tyler Bingham",
    "Tyler Davenport",
    "Tyler Villarreal",
    "Vinell Mariano",
    "Vinicius Rodrigues Pereira",
    "Vitor Manoel Dias",
    "Vitor Losnake",
    "Warley Oliveira da Silva",
    "Weston Hartman",
    "William Wright",
    "Winston Lopez",
    "Wyatt Rogers",
    "Zane Cook",
    "Riquelmi Santos",
    "Eric Novoa",
    "Boston Leather",
    "John Carlos Moreira",
    "Trey Holston"
    

]

# List of bulls to predict for (used if INPUT_NAME_FILE is missing)
BULL_NAMES = [
    "115 Night Agent",
    "165 Milestone",
    "64 Velvet Revolver",
    "723 Mike's Motive",
    "23 Toad",
    "035 Woody",
    "R62 UTZ BesTex Smokestack",
    # "106 Darkside",
    # "9 Eyes On Me",
    # "22-2 Nachos",
    # "50J Sweet Action",
    # "024 L.A.",
    # "114 I'm a Hostage",
    # "011 Boot Barn's Tyrone",
    # "247 Pegasus",
    # "J29 Greasy Bend",
    # "207 Wicked Solo",
    # "026 Let's Roll",
    # "2096 Sizmagraf",
    # "K25 Riser",
    # "963 Doze You Down",
    # "016 Fire Zone",
    # "953 July",
    # "A141 Pneu-Dart's Chief Wahoo",
    # "67F El Chapo",
    # "-17 Buffalo Heifer",
    # "2K Dirty Bomb",
    # "21 Lights Out",
    # "92 Hoobastank",
    # "0601 Let Him Fly",
    # "67 Benjamin Ranklin",
    # "55 J Lazy S",
    # "3H Landman",
    # "131 Feel the Magic",
    # "915 Bex Red Eye",
    # "9X Dark Thoughts",
    # "N11 The Kraken",
    # "G15 Snap Chatter",
    # "67H American Made",
    # "2165 Buck",
    # "012H Sunrise",
    # "7J Chicken In Black",
    # "67 American Hustle",
    # "666 Nefarious",
    # "24J Lieutenant Dan",
    # "H2-3 Always Been Crazy",
    # "110 Cherry Chew",
    # "19H Man Hater",
    # "H9 Holy Roller",
    # "F15 Socks In A Box",
    # "C43 Strapper",
    # "J38 Sour Patch",
    # "781 Uncle Rico",
    # "37H Electric Kitty",
    # "W800 Punchy Pete",
    # "36 Rockville",
    # "019 Old Testament",
    # "47J Vindicated",
    # "008 Rip",
    # "185 Noslaw Extrasauce",
    # "054 Levy",
    # "2015 Smack Ya",
    # "321 King Tut",
    # "955 Top Dollar",
    # "2162 Blonde Bomber",
    # "J44 Cherry Shot",
    # "38H- War Wagon",
    # "2246 Felix",
    # "128J The Player",
    # "94H Late Night Toss",
    # "320 Rock 'n Roll",
    # "15X Project X",
    # "82 Wingnut",
    # "143 Good Riddance",
    # "-205 Hermes",
    # "814 Tulsa Time",
    # "25-K Simple Man",
    # "73H Moving On",
    # "829 Ugly This",
    # "H15 Slick Rick",
    # "106 Scrappy",
    # "219 Cookie",
    # "101 Boot Barn's Skunk Kitty",
    # "G08 Smoke Down",
    # "716 Outlaw",
    # "948 Black Harbor",
    # "91 Washita Red",
    # "920 BRUNT Haymaker",
    # "626 Chateau Montelena's Montana Jacket",
    # "27J Yo Mamma",
    # "920 Snuggles",
    # "17J Torched",
    # "139 Foolish Pride",
    # "16K Fire Fight",
    # "000 Triple Aught",
    # "95J Alakazoo",
    # "8194 Blue Duck",
    # "K03 Murrell",
    # "40 Smokey",
    # "099 Shot Caller",
    # "36J Oyster Creek Brawler",
    # "31G Army Slasher",
    # "70H Stryker",
    # "113 Blue Nerve",
    # "813 Crazy Party",
    # "923 Ghost Face",
    # "6F Tchoupitoulas",
    # "60J Punchy",
    # "2105 Mouse Trap",
    # "X26 Real Deal",
    # "J39- Ransom",
    # "06 The Paint",
    # "17 Constant Payne",
    # "009 Margin of Error",
    # "121 Bam Bam",
    # "Y1 Tigger",
    # "918 Down Payment",
    # "113 High Caliber",
    # "14J Walk Hard",
    # "914 Mo Money",
    # "678Z Home Slice",
    # "89H Buck Nasty",
    # "048 Rank Hank",
    # "120 Smooth Violation",
    # "2127 Pay Dirt",
    # "906 In My Blood",
    # "116 Stacks",
    # "22J Average Joe",
    # "024 Wild Child",
    # "J740 Rafiki",
    # "053 Ares",
    # "919 Kodiac",
    # "40J Uncle Billy",
    # "1766 Power Trip",
    # "156 Ronnie Two Times",
    # "913 Windy",
    # "27 Hillbilly Heat",
    # "017 Swirl Your Glass",
    # "909 War Boy",
    # "017 OptiWize Mercy",
    # "766 Gaucho",
    # "2X Blackstone",
    # "901 The Gambler",
    # "58H Ruckus",
    # "-949 Puckered Up",
    # "715 Twisted Sister",
    # "4 Malibu's Top Dawg",
    # "925 Sky Walker",
    # "6X129 Iron Banana",
    # "748 The Jungle",
    # "914 Badlands",
    # "121 Judgement Day",
    # "13H Sucker Pop",
    # "61G Big Timber",
    # "07G Baldy",
    # "2020 Bruce",
    # "04 Mr Teller",
    # "31H Bullistic",
    # "56J Sherk",
    # "7003 Bueno",
    # "2001 Trash Panda",
    # "076 Toxic Masculinity",
    # "61 Redneck",
    # "81K Achilles",
    # "021 Gone Rogue",
    # "14-6 John 14:6",
    # "135 Itchy Scratchy",
    # "192 Snoop Dog",
    # "8455 Wild Card",
    # "49G Ah Hell",
    # "B10 Gravedigger",
    # "17G Hector",
    # "-18- Smackdown",
    # "I11 Moonlight Graham",
    # "977 Wild Fire",
    # "003 Slingin Tubs",
    # "P15 Play Time",
    # "894 Tijuana Two-Step",
    # "071H Apple Juicing",
    # "3371 Mustache Man",
    # "928 Finger Roll",
    # "2H Pepperoni",
    # "71 Kill Switch",
    # "76 Sneaky Situation",
    # "190 Horizons Creedmoor",
    # "498 Sharp Shooter",
    # "H604-1 Phantom",
    # "2016 Peanut",
    # "959 Lever Action",
    # "60/1 Wild Country",
    # "820 Feeds Red River",
    # "069 Snow Day",
    # "949 Smokin' Joe",
    # "1735 Boomerang",
    # "1747 Napoleon",
    # "2114 Easy Labor",
    # "2236 Blister",
    # "25H Nervous Hospital",
    # "197 Bookers Bandit",
    # "203 Caesar",
    # "2201 Loaded Question",
    # "938 Stone Cold ",
    # "115 Nine Lives",
    # "220 Bullicous",
    # "H27 Lift Me Up",
    # "834F Still Flyin' Crazy",
    # "K50 Elmo",
    # "161H Abracadabra",
    # "41K Go Blue",
    # "K46 Jailor",
    # "12K Uncle Adam",
    # "B-J65 Best Bet",
    # "04-K Shank",
    # "127 Lap Dancer",
    # "029 UB Smooth",
    # "0160 Level Up",
    # "B-080 Catching Gears",
    # "182 Black Tie",
    # "31J Body Bag",
    # "002 I Hate You",
    # "061 Lansky",
    # "B-931 Sugar Bear",
    # "909 Money Pit",
    # "18K Kippers Rippers",
    # "B-936 Mala Madre",
    # "119 Soul Man",
    # "83J Hold On Loosely",
    # "124 Holy Shift",
    # "128 Smores",
    # "128J Love County",
    # "182J Plowboy",
    # "2263 Mojo",
    # "-822 Air Marshall",
    # "B-035 New Dinero",
    # "004 Schliterbahn",
    # "28 Shy Pete",
    # "J02 Mike Honcho",
    # "824 EmmDee",
    # "102 Jeter",
    # "L7 Live Bait",
    # "211 Triton",
    # "405 Heavy Load",
    # "11 Triple Cross",
    # "-806 Lari's Speck",
    # "039 There's Your Sign",
    # "11k Wavy Bomb",
    # "035 Hay Ring",
    # "202K Boot Money",
    # "2003 Red Bull",
    # "960 Cash Goblin",
    # "720 Red Light",
    # "84J Fat Lip",
    # "029H What's Poppin",
    # "123 Short Fire",
    # "14 Interstate Daydream",
    # "7- Say When",
    # "033 Smokey",
    # "S827 Hoka Hey",
    # "53G Hard Candy",
    # "109 Purple Rain",
    # "H06 Dana White's Playmate",
    # "J79 Ridin' Dirty",
    # "MD-101 Boot Barn's Skunk Kitty",
    # "061 M.A.C.A.",
    # "-114 Night Prowler",
    # "104J Coach",
    # "916 Chico",
    # "021J Big Man",
    # "814 Boss Hoggin",
    # "001 PIF",
    # "1660 Nacho Night",
    # "2102 Big Jake",
    # "086 Brett's Silver Eagle",
    # "1H Fatal Attraction",
    # "185J Kiowa Havoc",
    # "821 Soul Train ",
    # "145 Magic Hunter",
    # "85J Peanut",
    # "020 Rowdy",
    # "189 Kung Fu Panda",
    # "106J Kevin",
    # "963J Big Bills",
    # "917 Peterbilt",
    # "904 Bomb Diggity",
    # "228Z Show Me The Money",
    # "129 Cramer",
    # "102J Hard Eight",
    # "RD915 Creek",
    # "55 Nickleback",
    # "9X5 Rip Tide",
    # "258 Float And Sting",
    # "164 Raider",
    # "030 Real Western ",
    # "927 Boot Barn's Sandman",
    # "G34 DirtyBru",
    # "12 Gunsmoke",
    # "147 Dozier",
    # "48J Last Call",
    # "727 Grey Fox",
    # "001 Nightmare",
    # "137 Demonic",
    # "95 Barbarosa",
    # "80 Whiplash",
    # "273 Rocket Man",
    # "1757 Another Round",
    # "980 Mighty Mouse",
    # "H04 Red Hot",
    # "639 The Undertaker",
    # "162 Drago",
    # "181 Turning Point",
    # "244 Nasubi",
    # "41J Mr. Jimmy",
    # "174 Squealin' Kitty",
    # "659 Slingin' It",
    # "819 American You",
    # "7 Pillow Talk",
    # "898 Simon",
    # "156 Trigger Happy",
    # "100 Mr. Koolie",
    # "1101 Wicked In a Winning Way",
    # "31 War Paint",
    # "1140 Karla's Pure Country",
    # "011 Designated Survivor",
    # "931 Big Red Train",
    # "1846 Diablo",
    # "38 38 Special ",
    # "925 Hunting Trip",
    # "C-527 Preachers Kid",
    # "110 Black Velvet",
    # "144 Blue Silhouette ",
    # "011 Jonah Hex",
    # "001 Boot Barn's Pack Rat",
    # "21-20 Mississippi Mercenary",
    # "2110 Hay Train",
    # "802 Vanilla Bean",
    # "103H I'll Make Ya Famous",
    # "1170 Big X",
    # "122J Josey Wales",
    # "8-9 Joe Cool",
    # "07 Night Fury",
    # "701 Magic Potion",
    # "2099 Muddy Jake",
    # "979 Fringe Minority",
    # "070 Cajun Brute",
    # "C-023 Living On A Prayer",
    # "32G Zapata",
    # "0816 Smoke Show",
    # "J17 Red Velvet",
    # "239 Sir Lion",
    # "005 Unhinged",
    # "2001 Mr. Man",
    # "4 Bucking Bill",
    # "103 Chili Mango",
    # "J1 Pearl Snap",
    # "988 Coal Train",
    # "19 Amarillo Highway",
    # "B2113 White Command",
    # "816 Bad Bob",
    # "2101 Question This",
    # "924 Red Dirt",
    # "194 Dixie Flash",
    # "2137 Dynamite",
    # "B82 Primo",
    # "123 True Grit",
    # "1257 Zorro",
    # "102 Cracky Chan",
    # "041 Big Red",
    # "X41 Hookie Monster",
    # "60J Cutting Edge Sugar Daddy",
    # "H10 Flossin",
    # "292 Squealer",
    # "1J The Intimidator",
    # "J2 Reality Check",
    # "065J Uncle Doc",
    # "9406 Honky Tonk",
    # "106 Not Like Us",
    # "70J Uncle Pete",
    # "411 Minds Broke",
    # "77 Cowboy Tuxedo",
    # "055 Santana",
    # "12-K Head Turner",
    # "102J Gold Buckle",
    # "-4 SPM",
    # "120J Pacifico",
    # "511 Holdin My Own",
    # "615 The Black",
    # "C363 Red Popcorn",
    # "1822 Fajita",
    # "29G Hard Core Slinger",
    # "106 Unhinged",
    # "804 Ace",
    # "8383 Dirty Dave",
    # "65F Lone Star",
    # "23 I'm Him",
    # "030 Whoa",
    # "015 Ezra The Great",
    # "258 Tropic Thunder",
    # "2041 Lock 'n Load",
    # "1726 Whip",
    # "05J Self Made",
    # "012 Crime Scene",
    # "24J Snappy",
    # "0498 Long Gone",
    # "177 Cutter",
    # "904 Cash On Black",
    # "C-922 Sly",
    # "2114 Where's the Beef",
    # "111 Mystified",
    # "0182 Workin' Man",
    # "J804 Fire Kat",
    # "17L Gucci Flip Flops",
    # "269 Red Ant",
    # "703 Hit or Miss",
    # "712 No Doubt",
    # "144 High Road",
    # "38J Blowhard",
    # "8X2 Dang, Dang",
    # "637D Border Crisis",
    # "176 Bosco",
    # "W150 Pay Pal",
    # "87H Tip It",
    # "910 Money Moves",
    # "90J Prince Charming",
    # "122 Lost In The Sauce",
    # "07 Irish Car Bomb",
    # "3G Frank",
    # "201 Organize Chaos",
    # "01 The One",
    # "S130 White Out",
    # "921 Johnny Rocket",
    # "003 Bad Boy",
    # "H09 Magic Boss",
    # "I17 Ridin' Salty",
    # "80G Jam Jam",
    # "8157 Skippy",
    # "9C Nobody",
    # "056 Sweet John",
    # "975 Reuben",
    # "33H- Little Country",
    # "497 Henry",
    # "041 Big Mac",
    # "693Y JL Gray's Ground Pounder",
    # "0021 Goin' Solo",
    # "808 Miller Time",
    # "16 Damaged Goods",
    # "161J Big Shasta",
    # "07 Bucktown",
    # "129 NashVegas",
    # "X71 Bad Intentions ",
    # "H07 Pac Man",
    # "04 Midnight Blues",
    # "15E Red Mosquito",
    # "704J Hot 'n Spicy",
    # "2015 Business Man",
    # "567 Body Roc",
    # "1904 Bone Digger",
    # "101 Sober Child",
    # "12 Rorschach",
    # "710 Do Dat Eddie",
    # "25g Bon Jovi",
    # "F51 Mr. Bojangles",
    # "16 Iceberg Slim",
    # "H94 Big Lean",
    # "956 Radical Dude",
    # "D10 Cooter Brown",
    # "008 Cockeye",
    # "034 Scooter Tooter",
    # "876 Swamp Donkey",
    # "209 Nakatosh Nation",
    # "877 Exodus",
    # "1211 Hail Mary",
    # "112 Halo",
    # "74 Jokers Misfit",
    # "20 TNT",
    # "H05 Calhoun Oil",
    # "007 The Don",
    # "68 Oreo",
    # "191 Barstool",
    # "379 Hey Jack",
    # "792 Red Headed Stranger",
    # "00 Gizmo",
    # "Y69 Bandito Bug",
    # "41 Spectacular",
    # "907 Dirty Deeds",
    # "1J Warlock",
    # "97 The Raven",
    # "066 Ruthless",
    # "8518 Pneu Dart's Gold Standard",
    # "014 Milestone",
    # "96 Bangarang",
    # "133H WinRocks Headache",
    # "145 Lazarus",
    # "569G Rogue One",
    # "930 Umm",
    # "109 Uncorked",
    # "891F Testified",
    # "932 Desert Twister",
    # "023 Gas Money",
    # "941 Buckaroo",
    # "14H Rombauer",
    # "122 Doctor Win",
    # "30H Happy Hour",
    # "P829 Rafter P Construction's Smooth Over It",
    # "16J Herded.com The Gambler",
    # "500 I'm Legit Too",
    # "301 Half Cocked",
    # "16 Pinecone",
    # "322 Show Me Dollars"
    ]

# Event date (defaults to config DEFAULT_DATE)
EVENT_DATE = DEFAULT_DATE

# Output file path
OUTPUT_FILE = "FA.xlsx"

# ---- Save arrays into Excel as requested ----
def _save_arrays_to_excel(xlsx_path: Path, riders: list[str], bulls: list[str]) -> None:
    try:
        xlsx_path.parent.mkdir(parents=True, exist_ok=True)
        mode = 'a' if xlsx_path.exists() else 'w'
        with pd.ExcelWriter(xlsx_path, engine='openpyxl', mode=mode, if_sheet_exists='replace') as writer:
            pd.DataFrame({"name": riders}).to_excel(writer, sheet_name='2025 Test', index=False)
            pd.DataFrame({"name": bulls}).to_excel(writer, sheet_name='Bulls', index=False)
        print(f"Wrote {len(riders)} riders to '2025 Test' and {len(bulls)} bulls to 'Bulls' in {xlsx_path}")
    except Exception as e:
        print(f"Could not write arrays to {xlsx_path}: {e}")

# ---- Load external names if provided ----
def _pick_first_existing_column(df: pd.DataFrame, options: list[str]) -> pd.Series | None:
    for col in options:
        if col in df.columns:
            return df[col]
    return None

def _read_sheet_any_name(xlsx_path: Path, candidates: list[str]) -> pd.DataFrame:
    xls = pd.ExcelFile(xlsx_path)
    available = {str(s).strip(): s for s in xls.sheet_names}
    for name in candidates:
        for k in available.keys():
            if k.lower() == str(name).lower().strip():
                return pd.read_excel(xlsx_path, sheet_name=available[k])
    return pd.read_excel(xlsx_path, sheet_name=next(iter(available.values())))

def _load_names_from_eos_excel(xlsx_path: Path) -> tuple[list[str] | None, list[str] | None]:
    try:
        rider_sheet_candidates = ["2025 Test", "2025 test", "2025_Test", "2025Test", "Riders", "riders"]
        riders_df = _read_sheet_any_name(xlsx_path, rider_sheet_candidates)
        riders_df.columns = [str(c).strip() for c in riders_df.columns]
        rider_series = _pick_first_existing_column(riders_df, [
            "name", "rider", "Rider_Name", "rider_name", "Rider", "Name", "RIDER", "NAME"
        ])
        if rider_series is None:
            upper_cols = [c for c in riders_df.columns if "upper" in str(c).lower()]
            lower_cols = [c for c in riders_df.columns if "lower" in str(c).lower()]
            if upper_cols:
                rider_series = riders_df[upper_cols[0]]
            elif lower_cols:
                rider_series = riders_df[lower_cols[0]]
            else:
                rider_series = riders_df.select_dtypes(include=["object"]).iloc[:, 0]
        riders = [str(n).strip() for n in rider_series.dropna().astype(str) if str(n).strip()]
        riders = list(dict.fromkeys(riders))

        bull_sheet_candidates = ["Bulls"]
        bulls_df = _read_sheet_any_name(xlsx_path, bull_sheet_candidates)
        bulls_df.columns = [str(c).strip() for c in bulls_df.columns]
        bull_series = _pick_first_existing_column(bulls_df, [
            "name", "bull", "bull_name", "Bull", "BULL", "Name", "NAME"
        ])
        if bull_series is None:
            bull_series = bulls_df.select_dtypes(include=["object"]).iloc[:, 0]
        bulls = [str(n).strip() for n in bull_series.dropna().astype(str) if str(n).strip()]
        bulls = list(dict.fromkeys(bulls))

        return riders, bulls
    except Exception as e:
        print(f"Failed to load lists from {xlsx_path}: {e}")
        return None, None

if __name__ == "__main__":
    # First, write the current arrays into the workbook exactly as requested
    _save_arrays_to_excel(INPUT_NAME_FILE, RIDER_NAMES, BULL_NAMES)

    # Prefer external list if available
    riders_from_file, bulls_from_file = (None, None)
    print(f"Looking for external list at: {INPUT_NAME_FILE}")
    if INPUT_NAME_FILE.exists():
        riders_from_file, bulls_from_file = _load_names_from_eos_excel(INPUT_NAME_FILE)
        if riders_from_file is not None:
            RIDER_NAMES = riders_from_file
        if bulls_from_file is not None:
            BULL_NAMES = bulls_from_file
        print(f"Loaded riders: {0 if RIDER_NAMES is None else len(RIDER_NAMES)} | bulls: {0 if BULL_NAMES is None else len(BULL_NAMES)}")
    else:
        print("External list not found; using in-script names.")

    print(f"=== Batch Prediction Script ===")
    print(f"Event Date: {EVENT_DATE}")
    print(f"Riders: {len(RIDER_NAMES)}")
    print(f"Bulls: {len(BULL_NAMES)}")
    print(f"Total Combinations: {len(RIDER_NAMES) * len(BULL_NAMES)}")
    print()

    # ---- RUN PREDICTIONS ----
    print("Running predictions...")
    print("Fast mode: 4-step pipeline (rider table, bull tables by hand, cartesian join, batch predict)")

    # Load once
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    from Predict.config import FINAL_DATA, FEATURE_LIST, MODEL_FILE, RIDER_XLSX, BULL_XLSX, RIDER_CSV
    from Predict.feature_engineering import solo_data_pull

    final_data = pd.read_csv(FINAL_DATA, parse_dates=["event_start_date"], low_memory=False)
    with open(FEATURE_LIST, encoding="utf-8") as fh:
        features = [ln.strip() for ln in fh if ln.strip()]
    bst = xgb.Booster()
    bst.load_model(str(MODEL_FILE))

    def _first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    # Check if final_data uses rider_internal_id (new pipeline) or rider_id (legacy)
    use_internal_id = "rider_internal_id" in final_data.columns
    
    # Rider ID map - prefer RIDER_XLSX (has rider_internal_id), fallback to RIDER_CSV
    rider_internal_dict = {}
    rider_legacy_dict = {}
    
    if RIDER_XLSX.exists():
        rid_df = pd.read_excel(RIDER_XLSX)
        rid_df.columns = [str(c).strip() for c in rid_df.columns]
        rider_name_col = _first_existing_col(rid_df, ["rider", "name", "rider_name", "Rider", "Name"])
        if rider_name_col is None:
            rider_name_col = rid_df.select_dtypes(include=["object"]).columns[0] if len(rid_df.select_dtypes(include=["object"]).columns) > 0 else rid_df.columns[0]
        
        # Try to get rider_internal_id first
        if "rider_internal_id" in rid_df.columns:
            rid_internal = rid_df[[rider_name_col, "rider_internal_id"]].copy()
            rid_internal["rider_internal_id"] = pd.to_numeric(rid_internal["rider_internal_id"], errors="coerce")
            rid_internal = rid_internal.dropna(subset=["rider_internal_id"])
            rider_internal_dict = {str(r).lower(): int(i) for r, i in zip(rid_internal[rider_name_col], rid_internal["rider_internal_id"]) if pd.notna(r) and pd.notna(i)}
        
        # Also get legacy rider_id if available
        if "rider_id" in rid_df.columns:
            rid_legacy = rid_df[[rider_name_col, "rider_id"]].copy()
            rid_legacy["rider_id"] = pd.to_numeric(rid_legacy["rider_id"], errors="coerce")
            rid_legacy = rid_legacy.dropna(subset=["rider_id"])
            rider_legacy_dict = {str(r).lower(): int(i) for r, i in zip(rid_legacy[rider_name_col], rid_legacy["rider_id"]) if pd.notna(r) and pd.notna(i)}
    
    # Fallback to RIDER_CSV if no mappings found
    if not rider_internal_dict and not rider_legacy_dict and RIDER_CSV.exists():
        rid_df = pd.read_csv(RIDER_CSV)
        if rid_df.shape[1] < 2:
            raise KeyError(f"Expected at least 2 columns in {RIDER_CSV}")
        rider_name_col = rid_df.columns[0]
        candidate_cols = list(rid_df.columns[1:])
        rider_id_col = candidate_cols[1] if len(candidate_cols) >= 2 else candidate_cols[0]
        coerced = pd.to_numeric(rid_df[rider_id_col], errors="coerce")
        if coerced.isna().mean() > 0.4 and len(candidate_cols) > 0:
            best_col = rider_id_col
            best_valid = (1 - coerced.isna().mean())
            for col in candidate_cols:
                c = pd.to_numeric(rid_df[col], errors="coerce")
                valid = (1 - c.isna().mean())
                if valid > best_valid:
                    best_col, best_valid, coerced = col, valid, c
            rider_id_col = best_col
        rid_map = rid_df[[rider_name_col, rider_id_col]].rename(columns={rider_name_col: "rider", rider_id_col: "rider_id"})
        rid_map["rider_id"] = pd.to_numeric(rid_map["rider_id"], errors="coerce")
        rid_map = rid_map.dropna(subset=["rider_id"])
        rider_legacy_dict = {str(r).lower(): int(i) for r, i in zip(rid_map["rider"], rid_map["rider_id"]) if pd.notna(r) and pd.notna(i)}
        print(f"Using rider ID column: '{rider_id_col}' from {RIDER_CSV}")
    
    # Debug: show mapping coverage
    if use_internal_id:
        print(f"Using rider_internal_id (new pipeline)")
        unmatched_riders = [r for r in RIDER_NAMES if rider_internal_dict.get(str(r).lower()) is None]
        print(f"Rider internal ID map entries: {len(rider_internal_dict)} | requested riders: {len(RIDER_NAMES)} | unmatched riders: {len(unmatched_riders)}")
    else:
        print(f"Using rider_id (legacy pipeline)")
        unmatched_riders = [r for r in RIDER_NAMES if rider_legacy_dict.get(str(r).lower()) is None]
        print(f"Rider ID map entries: {len(rider_legacy_dict)} | requested riders: {len(RIDER_NAMES)} | unmatched riders: {len(unmatched_riders)}")
    if unmatched_riders:
        print(f"Unmatched rider names (first 10): {unmatched_riders[:10]}")

    # Bull ID map from XLSX (first two columns: name, id)
    bid_df = pd.read_excel(BULL_XLSX)
    if bid_df.shape[1] < 2:
        raise KeyError(f"Expected at least 2 columns in {BULL_XLSX}")
    bull_name_col = bid_df.columns[0]
    bull_id_col   = bid_df.columns[1]
    bid_map = bid_df[[bull_name_col, bull_id_col]].rename(columns={bull_name_col: "bull", bull_id_col: "bull_id"})

    bull_dict  = {str(b).lower(): int(i) for b, i in zip(bid_map["bull"],  bid_map["bull_id"])  if pd.notna(b) and pd.notna(i)}
    unmatched_bulls = [b for b in set(BULL_NAMES) if bull_dict.get(str(b).lower()) is None]
    print(f"Bull ID map entries: {len(bull_dict)} | requested bulls: {len(set(BULL_NAMES))} | unmatched bulls: {len(unmatched_bulls)}")
    if unmatched_bulls:
        print(f"Unmatched bull names (first 10): {unmatched_bulls[:10]}")

    start_date = pd.to_datetime(EVENT_DATE)

    # Partition features
    rider_feature_cols = [c for c in features if c.startswith("r_") or c in ("new_rider_flag", "few_rides_flag", "few_rides_flag_hand")]
    bull_feature_cols  = [c for c in features if c.startswith("b_") or c.startswith("h_") or c == "new_bull_flag"]
    other_cols = [c for c in features if c not in rider_feature_cols and c not in bull_feature_cols]

    # Step 1: Rider table vs fixed bull "19H Man Hater"
    fixed_bull_name = "19H Man Hater"
    fixed_bid = bull_dict.get(str(fixed_bull_name).lower())
    if fixed_bid is None:
        print(f"Fixed bull '{fixed_bull_name}' not found in ID map. Exiting.")
        exit(1)

    rider_rows = {}
    rider_hand_map = {}
    for rider in RIDER_NAMES:
        key = str(rider).lower()
        rid_internal = rider_internal_dict.get(key) if use_internal_id else None
        rid_legacy = rider_legacy_dict.get(key) if not use_internal_id or rid_internal is None else None
        
        if rid_internal is None and rid_legacy is None:
            continue
        try:
            if rid_internal is not None:
                row = solo_data_pull(final_data, rider_id=None, bull_id=fixed_bid, rider_internal_id=rid_internal, start_date=start_date)
            else:
                row = solo_data_pull(final_data, rider_id=rid_legacy, bull_id=fixed_bid, rider_internal_id=None, start_date=start_date)
            for col in rider_feature_cols:
                if col not in row.columns:
                    row[col] = pd.NA
            rider_rows[rider] = row[rider_feature_cols].iloc[0]
            # Infer hand
            if rid_internal is not None:
                df_rider = final_data[final_data["rider_internal_id"] == rid_internal]
            else:
                df_rider = final_data[final_data["rider_id"] == rid_legacy]
            if not df_rider.empty and "hand" in df_rider.columns and df_rider["hand"].notna().any():
                try:
                    rider_hand_map[rider] = str(df_rider["hand"].value_counts().idxmax())
                except Exception:
                    rider_hand_map[rider] = "Unknown"
            else:
                rider_hand_map[rider] = "Unknown"
        except Exception as e:
            print(f"[rider_row] Failed for rider '{rider}': {e}")
            continue
    print(f"Built rider feature rows: {len(rider_rows)} (of {len(RIDER_NAMES)})")

    # Step 2: Bull tables vs fixed left/right riders
    fixed_left_rider = "Dalton Kasel"
    fixed_right_rider = "Kaique Pacheco"
    left_key = str(fixed_left_rider).lower()
    right_key = str(fixed_right_rider).lower()
    left_rid_internal = rider_internal_dict.get(left_key) if use_internal_id else None
    left_rid_legacy = rider_legacy_dict.get(left_key) if not use_internal_id or left_rid_internal is None else None
    right_rid_internal = rider_internal_dict.get(right_key) if use_internal_id else None
    right_rid_legacy = rider_legacy_dict.get(right_key) if not use_internal_id or right_rid_internal is None else None
    
    if (left_rid_internal is None and left_rid_legacy is None) or (right_rid_internal is None and right_rid_legacy is None):
        print("Fixed riders for left/right not found in ID map. Exiting.")
        exit(1)

    bull_rows_left = {}
    bull_rows_right = {}
    target_bulls = BULL_NAMES
    for bull in set(target_bulls):
        bid = bull_dict.get(str(bull).lower())
        if bid is None:
            continue
        try:
            if left_rid_internal is not None:
                row_l = solo_data_pull(final_data, rider_id=None, bull_id=bid, rider_internal_id=left_rid_internal, start_date=start_date)
            else:
                row_l = solo_data_pull(final_data, rider_id=left_rid_legacy, bull_id=bid, rider_internal_id=None, start_date=start_date)
            for col in bull_feature_cols:
                if col not in row_l.columns:
                    row_l[col] = pd.NA
            bull_rows_left[bull] = row_l[bull_feature_cols].iloc[0]
        except Exception as e:
            print(f"[bull_row L] Failed for bull '{bull}': {e}")
            pass
        try:
            if right_rid_internal is not None:
                row_r = solo_data_pull(final_data, rider_id=None, bull_id=bid, rider_internal_id=right_rid_internal, start_date=start_date)
            else:
                row_r = solo_data_pull(final_data, rider_id=right_rid_legacy, bull_id=bid, rider_internal_id=None, start_date=start_date)
            for col in bull_feature_cols:
                if col not in row_r.columns:
                    row_r[col] = pd.NA
            bull_rows_right[bull] = row_r[bull_feature_cols].iloc[0]
        except Exception as e:
            print(f"[bull_row R] Failed for bull '{bull}': {e}")
            pass
    print(f"Built bull feature rows: left={len(bull_rows_left)}, right={len(bull_rows_right)} (unique bulls requested={len(set(target_bulls))})")

    # Step 3: Cartesian product, select bull stats by rider hand
    combined_rows = []
    meta = []
    total_pairs = len(rider_rows) * len(target_bulls)
    built = 0
    for rider, rider_series in rider_rows.items():
        hand = str(rider_hand_map.get(rider, "Unknown")).lower()
        for bull in target_bulls:
            built += 1
            if built % 100 == 0:
                print(f"Composed {built}/{total_pairs} rows")
            bull_series = None
            if hand.startswith("l") and bull in bull_rows_left:
                bull_series = bull_rows_left[bull]
            elif hand.startswith("r") and bull in bull_rows_right:
                bull_series = bull_rows_right[bull]
            else:
                if bull in bull_rows_right:
                    bull_series = bull_rows_right[bull]
                elif bull in bull_rows_left:
                    bull_series = bull_rows_left[bull]
                else:
                    continue
            full = pd.Series(index=features, dtype=float)
            full[rider_feature_cols] = pd.to_numeric(rider_series.reindex(rider_feature_cols), errors="coerce")
            full[bull_feature_cols]  = pd.to_numeric(bull_series.reindex(bull_feature_cols), errors="coerce")
            combined_rows.append(full)
            meta.append((rider, rider_hand_map.get(rider, "Unknown"), bull))

    if not combined_rows:
        print("No composed rows to predict. Exiting.")
        # Extra debug context when nothing composes
        if len(rider_rows) == 0:
            print("Reason: No rider feature rows were built. Check rider name → ID mapping and data availability prior to event date.")
        if len(bull_rows_left) == 0 and len(bull_rows_right) == 0:
            print("Reason: No bull feature rows were built for either left or right fixed riders. Check bull name → ID mapping and data availability.")
        else:
            sample_rider = next(iter(rider_rows.keys()), None)
            if sample_rider is not None:
                hand = str(rider_hand_map.get(sample_rider, "Unknown"))
                print(f"Sample rider '{sample_rider}' inferred hand: {hand}")
        exit(0)

    X = pd.DataFrame(combined_rows)
    dm = xgb.DMatrix(X.values, feature_names=features)
    preds = bst.predict(dm)

    results_df = pd.DataFrame(meta, columns=["rider", "rider_hand", "bull"])\
        .assign(probability=np.round(preds.astype(float), 4))

    print(f"Generated {len(results_df)} predictions via composed rows.")
    print()

    # ---- CONVERT TO DATAFRAME ----
    df = results_df

    # Add percentage column for easier reading
    df['probability_pct'] = (df['probability'] * 100).round(2)

    # Sort by probability (highest first)
    df = df.sort_values('probability', ascending=False)

    # Reorder columns for better readability
    df = df[['rider', 'rider_hand', 'bull', 'probability', 'probability_pct']]

    # ---- PRINT SUMMARY ----
    print("=== PREDICTION SUMMARY ===")
    print(f"Top 5 Highest Probability Rides:")
    print(df.head().to_string(index=False))
    print()

    print(f"Bottom 5 Lowest Probability Rides:")
    print(df.tail().to_string(index=False))
    print()

    # Calculate some statistics
    avg_prob = df['probability'].mean()
    max_prob = df['probability'].max()
    min_prob = df['probability'].min()

    print(f"=== STATISTICS ===")
    print(f"Average Probability: {avg_prob:.4f} ({avg_prob*100:.2f}%)")
    print(f"Highest Probability: {max_prob:.4f} ({max_prob*100:.2f}%)")
    print(f"Lowest Probability:  {min_prob:.4f} ({min_prob*100:.2f}%)")
    print()

    # ---- SAVE TO EXCEL ----
    print(f"Saving results to {OUTPUT_FILE}...")

    with pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Predictions', index=False)
        summary_stats = pd.DataFrame({
            'Metric': ['Total Combinations', 'Average Probability', 'Highest Probability', 'Lowest Probability'],
            'Value': [len(df), f"{avg_prob:.4f}", f"{max_prob:.4f}", f"{min_prob:.4f}"],
            'Percentage': [f"{len(df)}", f"{avg_prob*100:.2f}%", f"{max_prob*100:.2f}%", f"{min_prob*100:.2f}%"]
        })
        summary_stats.to_excel(writer, sheet_name='Summary', index=False)
        df.head(10).to_excel(writer, sheet_name='Top 10 Rides', index=False)
        df.tail(10).to_excel(writer, sheet_name='Bottom 10 Rides', index=False)

    print(f"✓ Successfully saved to {OUTPUT_FILE}")
    print(f"✓ File contains {len(df)} predictions across {len(RIDER_NAMES)} riders and {len(BULL_NAMES)} bulls")
    print()
    print("Excel file contains the following sheets:")
    print("- Predictions: All predictions sorted by probability")
    print("- Summary: Key statistics and metrics")
    print("- Top 10 Rides: Highest probability combinations")
    print("- Bottom 10 Rides: Lowest probability combinations")
