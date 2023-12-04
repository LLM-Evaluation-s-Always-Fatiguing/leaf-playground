import asyncio
import json
import random
from typing import List, Optional

from pydantic import Field

from leaf_playground.core.scene import Scene, SceneConfig
from leaf_playground.data.log_body import LogBody
from leaf_playground.data.media import MediaType, Text
from leaf_playground.data.message import TextMessage
from leaf_playground.data.socket_data import SocketData, SocketDataType
from leaf_playground.zoo.who_is_the_spy.scene_agent import (
    PlayerDescription,
    PlayerPrediction,
    PlayerVote,
    ModeratorSummary,
    MessageTypes,
    Moderator,
    AIBasePlayer
)
from leaf_playground.zoo.who_is_the_spy.scene_info import (
    who_is_the_spy_scene_metadata,
    WhoIsTheSpySceneInfo,
    PlayerStatus
)


class WhoIsTheSpyLogBody(LogBody):
    references: Optional[List[MessageTypes]] = Field(default=None)
    response: MessageTypes = Field(default=...)


class WhoIsTheSpySceneConfig(SceneConfig):
    pass


class WhoIsTheSpyScene(Scene):
    config_obj = WhoIsTheSpySceneConfig
    config: config_obj

    metadata = who_is_the_spy_scene_metadata
    dynamic_agent_base_classes = [AIBasePlayer]
    scene_info_class = WhoIsTheSpySceneInfo
    log_body_class = WhoIsTheSpyLogBody

    def __init__(self, config: config_obj):
        super().__init__(config=config)

        self.moderator: Moderator = self.static_agents[0]
        self.players: List[AIBasePlayer] = self.agents

    async def _run(self):
        log_index = -1

        def put_message(message: MessageTypes):
            nonlocal log_index

            self.message_pool.put_message(message)

            log_index += 1
            references = None
            if not message.sender_id == self.moderator.id:
                references = self.message_pool.get_messages(message.sender)[:-1]
            self.socket_cache.append(
                SocketData(
                    type=SocketDataType.LOG,
                    data=self.log_body_class(
                        index=log_index,
                        references=references,
                        response=message,
                    ).model_dump(mode="json", by_alias=True)
                )
            )
            if isinstance(message, ModeratorSummary):
                self.socket_cache.append(
                    SocketData(
                        type=SocketDataType.SUMMARY,
                        data={
                            "type": message.summary_category.value,
                            "summary": message.summary
                        }
                    )
                )

        async def player_receive_key(player: AIBasePlayer) -> None:
            history = self.message_pool.get_messages(player.profile)
            key_assignment_msg = history[-1]
            try:
                await player.receive_key(key_assignment_msg)
            except:
                if self.config.debug_mode:
                    raise

        async def player_describe_key(player: AIBasePlayer) -> PlayerDescription:
            history = self.message_pool.get_messages(player.profile)
            try:
                description = await player.describe_key(
                    history, [self.moderator.profile] + [p.profile for p in players]
                )
            except:
                if self.config.debug_mode:
                    raise
                description = PlayerDescription(
                    sender=player.profile,
                    receivers=[self.moderator.profile] + [p.profile for p in players],
                    content=Text(text="I have nothing to say.")
                )
            return description

        async def player_describe_with_validation(player: AIBasePlayer) -> PlayerDescription:
            description = await player_describe_key(player)
            patience = 3
            while patience:
                valid, moderator_warning = self.moderator.valid_player_description(description=description)
                if valid:
                    break
                else:
                    put_message(moderator_warning)  # will be only seen by the player
                    description = await player_describe_key(player)
                patience -= 1
            return description

        async def players_describe_key(players_: List[AIBasePlayer]):
            for msg in self.moderator.ask_for_key_description(players=[p.profile for p in players_]):
                put_message(msg)
            players_descriptions = await asyncio.gather(
                *[player_describe_with_validation(player) for player in players_]
            )
            for description in players_descriptions:
                put_message(description)

        async def player_predict_role(player: AIBasePlayer) -> PlayerPrediction:
            history = self.message_pool.get_messages(player.profile)
            try:
                prediction = await player.predict_role(history, self.moderator.profile)
            except:
                if self.config.debug_mode:
                    raise
                prediction = PlayerPrediction(
                    sender=player.profile,
                    receivers=[self.moderator.profile],
                    content=Text(text=f"spy: [{player.name}]; blank: [{player.name}]")
                )
            return prediction

        async def player_vote(player: AIBasePlayer) -> PlayerVote:
            history = self.message_pool.get_messages(player.profile)
            try:
                vote = await player.vote(history, self.moderator.profile)
            except:
                if self.config.debug_mode:
                    raise
                vote = PlayerVote(
                    sender=player.profile,
                    receivers=[self.moderator.profile],
                    content=Text(text=f"vote: {player.profile}")
                )
            return vote

        num_rounds = self.scene_info.get_env_var("num_rounds").current_value
        while num_rounds:
            players = self.players
            random.shuffle(players)  # shuffle to randomize the speak order

            # clear information in the past round
            self.message_pool.clear()
            self.moderator.reset_inner_status()
            for player in players:
                player.reset_inner_status()

            # prepare the new game
            self.moderator.register_players(players=[player.profile for player in players])
            put_message(self.moderator.init_game())
            put_message(self.moderator.introduce_game_rule())
            put_message(self.moderator.announce_game_start())
            key_assignments, role_summarization = self.moderator.assign_keys()
            for msg in key_assignments:
                put_message(msg)
            put_message(role_summarization)
            await asyncio.gather(
                *[player_receive_key(player) for player in players]
            )

            # run game
            while True:  # for each turn
                # 1. ask players to give a description for the key they got sequentially,
                #    then validate player's prediction
                await players_describe_key(players)

                # 3. ask players to predict who is spy or blank slate
                for msg in self.moderator.ask_for_role_prediction():
                    put_message(msg)
                predictions = list(await asyncio.gather(*[player_predict_role(player) for player in players]))
                for prediction in predictions:
                    put_message(prediction)

                # 4. summarize player predictions
                put_message(self.moderator.summarize_players_prediction(predictions=predictions))

                patience = 3
                most_voted_players = None
                while patience:
                    # 5. ask players to vote
                    put_message(self.moderator.ask_for_vote())
                    votes = list(await asyncio.gather(*[player_vote(player) for player in players]))
                    for vote in votes:
                        put_message(vote)
                    # 6. summarize player votes, if there is a tie, ask most voted players to re-describe key
                    has_tie, vote_summarization, most_voted_players = self.moderator.summarize_players_votes(
                        votes=votes, focused_players=most_voted_players
                    )
                    put_message(vote_summarization)
                    if not has_tie:
                        break
                    # 7. most voted players re-describe
                    await players_describe_key([player for player in players if player.profile in most_voted_players])

                # 8. check is game over and announce winners
                game_over, winners_announcement = self.moderator.check_if_game_over()
                if game_over:
                    put_message(winners_announcement)
                    break
                # 9. exclude eliminated players
                players = [
                    player for player in players if self.moderator.player2status[player.id] == PlayerStatus.ALIVE
                ]

                # TODO: more things to do?

            num_rounds -= 1


__all__ = [
    "WhoIsTheSpySceneConfig",
    "WhoIsTheSpyScene"
]
