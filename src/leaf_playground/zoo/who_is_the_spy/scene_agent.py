import random
from abc import abstractmethod
from enum import Enum
from inspect import Signature, Parameter
from itertools import chain
from typing import Dict, List, Optional, Tuple, Set, Union
from uuid import UUID

from pydantic import Field

from leaf_playground.core.scene_agent import SceneAIAgent, SceneAIAgentConfig, SceneStaticAgent, SceneStaticAgentConfig
from leaf_playground.core.scene_info import SceneInfo
from leaf_playground.data.message import TextMessage
from leaf_playground.data.media import Audio, Image, Text
from leaf_playground.data.profile import Profile, Role
from leaf_playground.utils.import_util import DynamicObject
from leaf_playground.zoo.who_is_the_spy.scene_info import PlayerRoles, PlayerStatus, KeyModalities, WhoIsTheSpySceneInfo
from leaf_playground.zoo.who_is_the_spy.text_utils import get_most_similar_text


KEY_PLACEHOLDER = "<KEY>"


class PlayerDescription(TextMessage):
    pass


class PlayerPrediction(TextMessage):
    def get_prediction(self, player_names: List[str], has_blank_slate: bool) -> Dict[PlayerRoles, Set[str]]:
        def retrieve_names(symbol: str) -> Set[str]:
            names = set()
            content = self.content.text
            if symbol in names:
                content = content[content.index(symbol) + len(symbol):].strip()
                content = content.split(":")[0].strip()
                for pred in content.split(","):
                    pred = pred.strip()
                    names.add(get_most_similar_text(pred, [each.strip() for each in player_names]))
            return names

        preds = {PlayerRoles.SPY: retrieve_names("spy:")}
        if has_blank_slate:
            preds[PlayerRoles.BLANK_SLATE] = retrieve_names("blank:")
        return preds


class PlayerVote(TextMessage):
    def get_vote(self, player_names: List[str]) -> str:
        vote = self.content.text
        get_vote = False
        if "vote:" in vote:
            vote = vote[vote.index("vote:") + len("vote:"):].strip()
            vote = get_most_similar_text(vote, [each.strip() for each in player_names])
            get_vote = True

        return vote if get_vote else ""


class ModeratorKeyAssignment(TextMessage):
    key: Optional[Union[Audio, Image, Text]] = Field(default=None)

    @classmethod
    def create_with_key(cls, key: Union[Audio, Image, Text], sender: Profile, receiver: Profile):
        return cls(
            sender=sender,
            receivers=[receiver],
            content=Text(text=f"{receiver.name}, your key is: {KEY_PLACEHOLDER}"),
            key=key
        )

    @classmethod
    def create_without_key(cls, sender: Profile, receiver: Profile):
        return cls(
            sender=sender,
            receivers=[receiver],
            content=Text(text=f"{receiver.name}, you got a blank clue."),
            key=None
        )


class ModeratorAskForDescription(TextMessage):
    @classmethod
    def create(cls, sender: Profile, receiver: Profile) -> "ModeratorAskForDescription":
        return cls(
            sender=sender,
            receivers=[receiver],
            content=Text(
                text=f"{receiver.name}, please using ONE-sentence to describe your key."
            )
        )


class ModeratorAskForRolePrediction(TextMessage):
    @classmethod
    def create(
        cls,
        sender: Profile,
        receiver: Profile,
        player_names: List[str],
        has_blank_slate: bool
    ) -> "ModeratorAskForRolePrediction":
        if has_blank_slate:
            msg = (
                f"{receiver.name}, now think about who is the spy and who is the blank slate. "
                f"And tell me your predictions in the following format:\n"
                "spy: [<player_name>, ..., <player_name>]; blank: [<player_name>, ..., <player_name>]<EOS>\n"
                "Where <player_name> is the name of the player you think is the spy or the blank slate.\n"
            )
        else:
            msg = (
                f"{receiver.name}, now think about who is the spy. "
                f"And tell me your predictions in the following format:\n"
                f"spy: [<player_name>, ..., <player_name>]<EOS>\n"
                f"Where <player_name> is the name of the player you think is the spy.\n"
            )
        return cls(
            sender=sender,
            receivers=[receiver],
            content=Text(
                text=msg + f"Player names are: {player_names}.\nYour response MUST starts with 'spy:'"
            )
        )


class ModeratorAskForVote(TextMessage):
    @classmethod
    def create(
        cls,
        sender: Profile,
        receivers: List[Profile],
        has_blank_slate: bool
    ) -> "ModeratorAskForVote":
        if has_blank_slate:
            msg = (
                "And now，let's vote for who should be eliminated. "
                "Civilians vote for who they suspect is the Spy or Blank Slate. "
                "Spies vote for a likely Civilian or Blank Slate. "
                "Blank Slates vote for their suspected Spy. "
                "Each person can only cast one vote and cannot vote for themselves, "
                "please send me your vote in the following format:\n"
                "vote: <player_name><EOS>\n"
                "Where <player_name> is the name of the player you want to vote for.\n"
            )
        else:
            msg = (
                "And now, let's vote for who should be eliminated. "
                "Civilians vote for who they suspect is the Spy. "
                "Spies vote for a likely Civilian. "
                "Each person can only cast one vote and cannot vote for themselves, "
                "please send me your vote in the following format:\n"
                "vote: <player_name><EOS>\n"
                "Where <player_name> is the name of the player you want to vote for.\n"
            )
        return cls(
            sender=sender,
            receivers=[player for player in receivers],
            content=Text(
                text=msg + f"Player names are: {','.join([player.name for player in receivers])}.\n"
                           "Your response MUST starts with 'vote:'"
            )
        )


class SummaryCategory(Enum):
    KEYS = "keys"
    ROLES = "roles"
    ROLE_PREDICTION = "role_prediction"
    VOTE = "vote"
    WINNER = "winner"


class ModeratorSummary(TextMessage):
    summary_category: SummaryCategory = Field(default=...)
    summary: dict = Field(default=...)


class ModeratorWarning(TextMessage):
    pass


MessageTypes = Union[
    TextMessage,
    PlayerDescription,
    PlayerPrediction,
    PlayerVote,
    ModeratorAskForDescription,
    ModeratorAskForRolePrediction,
    ModeratorAskForVote,
    ModeratorSummary,
    ModeratorWarning
]


KEYS = {  # TODO: this is a temporarily mock, implement data loading mechanism in dataset_utils module
    KeyModalities.TEXT: [
        ("apple", "pear"),
        ("pen", "pencil"),
        ("milk", "soya-bean milk"),
        ("car", "bus"),
        ("cat", "dog"),
        ("book", "magazine"),
        ("computer", "phone"),
        ("table", "chair"),
        ("window", "door"),
        ("bed", "sofa"),
        ("bread", "cake"),
        ("chocolate", "candy"),
        ("pizza", "hamburger"),
        ("chicken", "duck"),
    ]
}


class AIBasePlayerConfig(SceneAIAgentConfig):
    context_max_tokens: int = Field(default=4096)


class AIBasePlayer(SceneAIAgent):
    config_obj = AIBasePlayerConfig
    config: config_obj

    _actions: Dict[str, Signature] = {
        "receive_key": Signature(
            parameters=[
                Parameter(
                    name="key_assignment_message",
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=ModeratorKeyAssignment
                )
            ],
            return_annotation=None
        ),
        "describe_key": Signature(
            parameters=[
                Parameter(
                    name="history",
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=List[MessageTypes]
                ),
                Parameter(
                    name="receivers",
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=List[Profile]
                )
            ],
            return_annotation=PlayerDescription
        ),
        "predict_role": Signature(
            parameters=[
                Parameter(
                    name="history",
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=List[MessageTypes]
                ),
                Parameter(
                    name="moderator",
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=Profile
                )
            ],
            return_annotation=PlayerPrediction
        ),
        "vote": Signature(
            parameters=[
                Parameter(
                    name="history",
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=List[MessageTypes]
                ),
                Parameter(
                    name="moderator",
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=Profile
                )
            ],
            return_annotation=PlayerVote
        ),
        "reset_inner_status": Signature(
            parameters=None,
            return_annotation=None
        )
    }

    def __init__(self, config: config_obj):
        super().__init__(config=config)

    def reset_inner_status(self) -> None:
        pass

    @abstractmethod
    async def receive_key(self, key_assignment_message: ModeratorKeyAssignment) -> None:
        pass

    @abstractmethod
    async def describe_key(self, history: List[MessageTypes], receivers: List[Profile]) -> PlayerDescription:
        pass

    @abstractmethod
    async def predict_role(self, history: List[MessageTypes], moderator: Profile) -> PlayerPrediction:
        pass

    @abstractmethod
    async def vote(self, history: List[MessageTypes], moderator: Profile) -> PlayerVote:
        pass


class ModeratorConfig(SceneStaticAgentConfig):
    profile: Profile = Field(default=Profile(name="Moderator"))


class Moderator(SceneStaticAgent):
    config_obj = ModeratorConfig
    config: config_obj

    scene_info: WhoIsTheSpySceneInfo

    _actions: Dict[str, Signature] = {
        "register_players": Signature(
            parameters=[
                Parameter(
                    name="players",
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=List[Profile]
                )
            ],
            return_annotation=None
        ),
        "init_game": Signature(
            parameters=None,
            return_annotation=ModeratorSummary
        ),
        "introduce_game_rule": Signature(
            parameters=None,
            return_annotation=TextMessage
        ),
        "announce_game_start": Signature(
            parameters=None,
            return_annotation=TextMessage
        ),
        "assign_keys": Signature(
            parameters=None,
            return_annotation=Tuple[List[ModeratorKeyAssignment], ModeratorSummary]
        ),
        "ask_for_key_description": Signature(
            parameters=[
                Parameter(
                    name="players",
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=List[Profile]
                )
            ],
            return_annotation=List[ModeratorAskForDescription]
        ),
        "valid_player_description": Signature(
            parameters=[
                Parameter(
                    name="description",
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=PlayerDescription
                )
            ],
            return_annotation=Tuple[bool, Optional[ModeratorWarning]]
        ),
        "ask_for_role_prediction": Signature(
            parameters=None,
            return_annotation=List[ModeratorAskForRolePrediction]
        ),
        "summarize_players_prediction": Signature(
            parameters=[
                Parameter(
                    name="predictions",
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=List[PlayerPrediction]
                )
            ],
            return_annotation=ModeratorSummary
        ),
        "ask_for_vote": Signature(
            parameters=None,
            return_annotation=ModeratorAskForVote
        ),
        "summarize_players_votes": Signature(
            parameters=[
                Parameter(
                    name="votes",
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=List[PlayerVote]
                ),
                Parameter(
                    name="focused_players",
                    kind=Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                    annotation=Optional[List[Profile]]
                )
            ],
            return_annotation=Tuple[bool, ModeratorSummary, Optional[List[Profile]]]
        ),
        "check_if_game_over": Signature(
            parameters=None,
            return_annotation=Tuple[bool, Optional[ModeratorSummary]]
        ),
        "reset_inner_status": Signature(
            parameters=None,
            return_annotation=None
        )
    }

    description: str = "A static agent that moderates the game."
    obj_for_import: DynamicObject = DynamicObject(
        obj="Moderator",
        module="leaf_playground.zoo.who_is_the_spy.scene_agent"
    )

    game_rule_with_blank = (
        "You are playing a game of Who is the spy. Here are the game rules:\n\n"
        "## Information and roles\n\n"
        "There are three roles in \"Who is the Spy?\": Spy, Civilian, and Blank Slate.\n"
        "- Civilians are shown the correct key.\n"
        "- Spies see a key similar to the correct one but incorrect.\n"
        "- Blank Slates receive a blank clue.\n"
        "Your role is unknown to you, so careful listening and inference are crucial to identify the spy.\n\n"
        "## Objectives\n\n"
        "Your objectives vary based on your role:\n"
        "- As a Civilian, your aim is to identify and vote out the Spy and the Blank Slate, without revealing the "
        "correct key. Focus first on finding the Blank Slate.\n"
        "- If you're the Spy, your goal is to blend in, avoid detection, and survive the voting. Winning occurs if "
        "at least one Spy remains at the end.\n"
        "- As a Blank Slate, try to uncover and vote out the Spy without revealing your own role. You can guess and "
        "describe what you think is the correct key.\n\n"
        "## Stages\n\n"
        "The game has two main stages and one special scenario:\n"
        "1. Giving Clues Stage: Each player gives clues about their key. Blank Slates can describe anything "
        "they choose.\n"
        "2. Accusation Stage: Here, Civilians vote for who they suspect is the Spy or Blank Slate. Spies vote "
        "for a likely Civilian or Blank Slate. Blank Slates vote for their suspected Spy.\n"
        "3. Tiebreaker Scenario: In the event of a tie, those with the most votes will re-describe their key, "
        "and a new vote takes place among them.\n\n"
        "## Code of Conduct\n\n"
        "Here are five rules of behavior you need to follow:\n"
        "- Your clues should be brief and not include the key.\n"
        "- Your clues can't duplicate the previous one.\n"
        "- Do not pretend you are other players or the moderator.\n"
        "- You cannot vote for yourself.\n"
        "- Always end your response with <EOS>."
    )
    game_rule_without_blank = (
        "You are playing a game of Who is the spy. Here are the game rules:\n\n"
        "## Information and roles\n\n"
        "There are two roles in \"Who is the Spy?\": Spy and Civilian.\n"
        "- Civilians are shown the correct key.\n"
        "- Spies see a key similar to the correct one but incorrect.\n"
        "Your role is unknown to you, so careful listening and inference are crucial to identify the spy.\n\n"
        "## Objectives\n\n"
        "Your objectives vary based on your role:\n"
        "- As a Civilian, your aim is to identify and vote out the Spy, without revealing the correct key.\n"
        "- If you're the Spy, your goal is to blend in, avoid detection, and survive the voting. Winning occurs "
        "if at least one Spy remains at the end.\n\n"
        "## Stages\n\n"
        "The game has two main stages and one special scenario:\n"
        "1. Giving Clues Stage: Each player gives clues about their key.\n"
        "2. Accusation Stage: Here, Civilians vote for who they suspect is the Spy or Blank Slate.Spies vote for "
        "a likely Civilian or Blank Slate.\n"
        "3. Tiebreaker Scenario: In the event of a tie, those with the most votes will re-describe their key, and "
        "a new vote takes place among them.\n\n"
        "## Code of Conduct\n\n"
        "Here are five rules of behavior you need to follow:\n"
        "- Your clues should be brief and not include the key.\n"
        "- Your clues can't duplicate the previous one.\n"
        "- Do not pretend you are other players or the moderator.\n"
        "- You cannot vote for yourself.\n"
        "- Always end your response with <EOS>."
    )

    def __init__(self, config: config_obj):
        super().__init__(config=config)

        self.player2role: Dict[UUID, PlayerRoles] = {}
        self.role2players: Dict[PlayerRoles, List[Profile]] = {
            PlayerRoles.CIVILIAN: [],
            PlayerRoles.SPY: [],
            PlayerRoles.BLANK_SLATE: []
        }
        self.id2player: Dict[UUID, Profile] = {}
        self.player2status: Dict[UUID, PlayerStatus] = {}
        self.civilian_key: Union[Audio, Image, Text] = None
        self.spy_key: Union[Audio, Image, Text] = None

    def post_init(self, role: Optional[Role], scene_info: SceneInfo):
        super().post_init(role, scene_info)
        self.has_blank_slate = self.scene_info.get_env_var("has_blank_slate").current_value
        self.roles_assignment_strategy = {
            4: {
                PlayerRoles.CIVILIAN: 3,
                PlayerRoles.SPY: 1,
                PlayerRoles.BLANK_SLATE: 0
            },
            5: {
                PlayerRoles.CIVILIAN: 3 if self.has_blank_slate else 4,
                PlayerRoles.SPY: 1,
                PlayerRoles.BLANK_SLATE: 1 if self.has_blank_slate else 0
            },
            6: {
                PlayerRoles.CIVILIAN: 4 if self.has_blank_slate else 5,
                PlayerRoles.SPY: 1,
                PlayerRoles.BLANK_SLATE: 1 if self.has_blank_slate else 0
            },
            7: {
                PlayerRoles.CIVILIAN: 4 if self.has_blank_slate else 5,
                PlayerRoles.SPY: 2,
                PlayerRoles.BLANK_SLATE: 1 if self.has_blank_slate else 0
            },
            8: {
                PlayerRoles.CIVILIAN: 5 if self.has_blank_slate else 6,
                PlayerRoles.SPY: 2,
                PlayerRoles.BLANK_SLATE: 1 if self.has_blank_slate else 0
            },
            9: {
                PlayerRoles.CIVILIAN: 6 if self.has_blank_slate else 7,
                PlayerRoles.SPY: 2,
                PlayerRoles.BLANK_SLATE: 1 if self.has_blank_slate else 0
            }
        }

    def register_players(self, players: List[Profile]) -> None:
        for player in players:
            self.id2player[player.id] = player
            self.player2status[player.id] = PlayerStatus.ALIVE

    def init_game(self) -> ModeratorSummary:
        num_players = len(self.id2player)
        roles_agent_num = self.roles_assignment_strategy[num_players]

        roles = list(chain(*[[role] * agent_num for role, agent_num in roles_agent_num.items()]))
        random.shuffle(roles)  # shuffle to randomize the role assignment
        for player_id, role in zip(list(self.id2player.keys()), roles):
            self.role2players[role].append(self.id2player[player_id])

        keys = list(random.choice(KEYS[self.scene_info.get_env_var("key_modality").current_value]))
        random.shuffle(keys)  # shuffle to randomize the key assignment
        self.civilian_key, self.spy_key = Text(text=keys[0]), Text(text=keys[1])

        return ModeratorSummary(
            sender=self.profile,
            receivers=[self.profile],
            content=Text(
                text=f"{PlayerRoles.CIVILIAN.value}_key is {self.civilian_key}, "
                     f"{PlayerRoles.SPY.value}_key is {self.spy_key}."
            ),
            summary_category=SummaryCategory.KEYS,
            summary={
                f"{PlayerRoles.CIVILIAN.value}_key": self.civilian_key,
                f"{PlayerRoles.SPY.value}_key": self.spy_key
            }
        )

    def introduce_game_rule(self) -> TextMessage:
        msg = self.game_rule_with_blank if self.has_blank_slate else self.game_rule_without_blank
        return TextMessage(
            sender=self.profile,
            receivers=[player for player in self.id2player.values()],
            content=Text(text=msg)
        )

    def announce_game_start(self) -> TextMessage:
        num_players = len(self.id2player)
        role2word = {
            PlayerRoles.CIVILIAN: "civilians",
            PlayerRoles.SPY: "spies",
            PlayerRoles.BLANK_SLATE: "blank slates"
        }
        roles_num_description = ", ".join(
            [f"{len(role_players)} {role2word[role]}" for role, role_players in self.role2players.items()]
        )

        return TextMessage(
            sender=self.profile,
            receivers=[player for player in self.id2player.values()],
            content=Text(
                text=f"Now the game begins! There are {num_players} players in this game, including "
                     f"{roles_num_description}."
            )
        )

    def assign_keys(self) -> Tuple[List[ModeratorKeyAssignment], ModeratorSummary]:
        messages = []
        for role, players in self.role2players.items():
            for player in players:
                self.player2role[player.id] = role
                if role == PlayerRoles.CIVILIAN:
                    msg = ModeratorKeyAssignment.create_with_key(
                        key=self.civilian_key, sender=self.profile, receiver=player
                    )
                elif role == PlayerRoles.SPY:
                    msg = ModeratorKeyAssignment.create_with_key(
                        key=self.spy_key, sender=self.profile, receiver=player
                    )
                else:
                    msg = ModeratorKeyAssignment.create_without_key(sender=self.profile, receiver=player)
                messages.append(msg)

        return messages, ModeratorSummary(
            sender=self.profile,
            receivers=[self.profile],
            content=Text(
                text="\n".join(
                    [
                        f"{role.value}: {[player.name for player in players]}"
                        for role, players in self.role2players.items()
                    ]
                )
            ),
            summary_category=SummaryCategory.ROLES,
            summary={
                role.value: [player.name for player in players] for role, players in self.role2players.items()
            }
        )

    def ask_for_key_description(self, players: List[Profile]) -> List[ModeratorAskForDescription]:
        return [
            ModeratorAskForDescription.create(
                sender=self.profile,
                receiver=player,
            ) for player in players if self.player2status[player.id] == PlayerStatus.ALIVE
        ]

    def valid_player_description(self, description: PlayerDescription) -> Tuple[bool, Optional[ModeratorWarning]]:
        sender_id = description.sender.id
        sender_role = self.player2role[sender_id]
        if sender_role == PlayerRoles.BLANK_SLATE:
            return True, None
        if self.scene_info.get_env_var("key_modality") == KeyModalities.TEXT:
            warn_msg = "Your description contains your key, which is not allowed, please redo the description."
            if sender_role == PlayerRoles.CIVILIAN and self.civilian_key.text in description.content.text:
                return False, ModeratorWarning(
                    sender=self.profile,
                    receivers=[description.sender],
                    content=Text(text=warn_msg)
                )
            if sender_role == PlayerRoles.SPY and self.spy_key.text in description.content.text:
                return False, ModeratorWarning(
                    sender=self.profile,
                    receivers=[description.sender],
                    content=Text(text=warn_msg)
                )

        return True, None

    def ask_for_role_prediction(self) -> List[ModeratorAskForRolePrediction]:
        player_names = [
            player.name for player in self.id2player.values() if self.player2status[player.id] == PlayerStatus.ALIVE
        ]
        return [
            ModeratorAskForRolePrediction.create(
                sender=self.profile,
                receiver=player,
                player_names=player_names,
                has_blank_slate=self.has_blank_slate
            ) for player in self.id2player.values() if self.player2status[player.id] == PlayerStatus.ALIVE
        ]

    def summarize_players_prediction(self, predictions: List[PlayerPrediction]) -> ModeratorSummary:
        player2res = {}
        for prediction in predictions:
            player_id = prediction.sender_id
            preds = prediction.get_prediction(
                player_names=[player.name for player in self.id2player.values()],
                has_blank_slate=self.has_blank_slate
            )
            accuracy = sum(
                [set([p.id for p in self.role2players[role]]) == [p.id for p in preds[role]] for role in preds]
            ) / len(preds)

            player2res[player_id] = {
                "prediction": {role.value: list(names) for role, names in preds.items()},
                "label": {role.value: self.role2players[role] for role in preds.keys()},
                "accuracy": accuracy
            }
        return ModeratorSummary(
            sender=self.profile,
            receivers=[self.profile],
            content=Text(text=""),
            summary_category=SummaryCategory.ROLE_PREDICTION,
            summary=player2res
        )

    def ask_for_vote(self) -> ModeratorAskForVote:
        return ModeratorAskForVote.create(
            sender=self.profile,
            receivers=[
                player for player in self.id2player.values() if self.player2status[player.id] == PlayerStatus.ALIVE
            ],
            has_blank_slate=self.has_blank_slate
        )

    def summarize_players_votes(
        self,
        votes: List[PlayerVote],
        focused_players: Optional[List[Profile]] = None
    ) -> Tuple[bool, ModeratorSummary, Optional[List[Profile]]]:

        def get_most_voted_players() -> List[Profile]:
            eliminated_names = [
                player_name for player_name, num_be_voted in player2num_be_voted.items() if
                num_be_voted == max(player2num_be_voted.values())
            ]
            return [player for player in self.id2player.values() if player.name in eliminated_names]

        player2num_be_voted = {player.name: 0 for player in self.id2player.values()}
        player2votes = {}
        for vote in votes:
            vote_to = vote.get_vote([player.name for player in self.id2player.values()])
            if not vote_to:
                vote_to = vote.sender_name  # set to the voter as a punishment for not voting
            player2votes[vote.sender_name] = vote_to
            player2num_be_voted[vote_to] += 1
        if focused_players:
            focused_names = [p.name for p in focused_players]
            for player_name in player2num_be_voted:
                if player_name not in focused_names:
                    player2num_be_voted[player_name] = 0

        voting_detail = "\n".join([f"{voter} votes to {voted}" for voter, voted in player2votes.items()]) + "\n"
        if focused_players:
            voting_detail += (
                f"This is a re-voting turn, we will only focus on the votes {[p.name for p in focused_players]} got.\n"
            )
        most_voted_players = get_most_voted_players()
        if len(most_voted_players) > 1:  # tied
            return (
                True,
                ModeratorSummary(
                    sender=self.profile,
                    receivers=[player for player in self.id2player.values()],
                    content=Text(
                        text=f"{voting_detail}{[p.name for p in most_voted_players]} are having the same "
                             f"votes, for those players, please re-describe the key you received again."
                    ),
                    summary_category=SummaryCategory.VOTE,
                    summary={"tied": [p.model_dump(mode="json") for p in most_voted_players]}
                )
                ,
                most_voted_players
            )
        else:  # eliminate
            for player in most_voted_players:
                self.player2status[player.id] = PlayerStatus.ELIMINATED
            return (
                False,
                ModeratorSummary(
                    sender=self.profile,
                    receivers=[player for player in self.id2player.values()],
                    content=Text(
                        text=f"{voting_detail}{most_voted_players[0].name} has the most votes and is eliminated."
                    ),
                    summary_category=SummaryCategory.VOTE,
                    summary={"eliminated": [loser.model_dump(mode="json") for loser in most_voted_players]}
                ),
                None
            )

    def check_if_game_over(self) -> Tuple[bool, Optional[ModeratorSummary]]:
        def return_game_over(role: PlayerRoles):
            winners = [
                player.name for player in self.role2players[role]
                if self.player2status[player.id] == PlayerStatus.ALIVE
            ]
            return True, ModeratorSummary(
                sender=self.profile,
                receivers=[player for player in self.id2player.values()],
                content=Text(text=f"Game Over! {role.value} win, winners are: {winners}."),
                summary_category=SummaryCategory.WINNER,
                summary={"winner_role": role.value, "winners": winners}
            )

        num_players = len(self.id2player)
        num_alive_players = len(
            [player for player, status in self.player2status.items() if status == PlayerStatus.ALIVE]
        )
        num_alive_civilians = len(
            [
                player for player in self.role2players[PlayerRoles.CIVILIAN]
                if self.player2status[player.id] == PlayerStatus.ALIVE
            ]
        )
        num_alive_spies = len(
            [
                player for player in self.role2players[PlayerRoles.SPY]
                if self.player2status[player.id] == PlayerStatus.ALIVE
            ]
        )
        if num_alive_civilians == num_alive_players:  # civilians win
            return return_game_over(PlayerRoles.CIVILIAN)
        if (
            (num_players > 6 and num_alive_players <= 3 and num_alive_spies > 0) or
            (num_players <= 6 and num_alive_players <= 2 and num_alive_spies > 0)
        ):  # spies win
            return return_game_over(PlayerRoles.SPY)
        if self.has_blank_slate and num_alive_spies == 0 and num_alive_civilians != num_alive_players:  # blank wins
            return return_game_over(PlayerRoles.BLANK_SLATE)
        return False, None

    def reset_inner_status(self) -> None:
        self.player2role: Dict[UUID, PlayerRoles] = {}
        self.role2players: Dict[PlayerRoles, List[Profile]] = {
            PlayerRoles.CIVILIAN: [],
            PlayerRoles.SPY: [],
            PlayerRoles.BLANK_SLATE: []
        }
        self.id2player: Dict[UUID, Profile] = {}
        self.player2status: Dict[UUID, PlayerStatus] = {}
        self.civilian_key: Union[Audio, Image, Text] = None
        self.spy_key: Union[Audio, Image, Text] = None


__all__ = [
    "KEY_PLACEHOLDER",
    "PlayerDescription",
    "PlayerPrediction",
    "PlayerVote",
    "ModeratorKeyAssignment",
    "ModeratorAskForDescription",
    "ModeratorAskForRolePrediction",
    "ModeratorAskForVote",
    "ModeratorSummary",
    "ModeratorWarning",
    "MessageTypes",
    "ModeratorConfig",
    "Moderator",
    "AIBasePlayerConfig",
    "AIBasePlayer"
]
