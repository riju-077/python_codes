from importer.amzn_api_interaction import (
    SponsoredBrands,
    SponsoredProducts,
    SponsoredDisplays,
)
from importer.querycute import Querycute
from importer.PSTtime import *
from new_deep_module import *
from collections import defaultdict
import numpy as np
from sklearn.preprocessing import StandardScaler

sb = SponsoredBrands()
sp = SponsoredProducts()
sd = SponsoredDisplays()
qcm = Querycute("modeling_db")


minute_from_last_reset = 0


path_dict = {
    "SP_BID": {
        "policy_model_path": f"./{running_model_name('SP_BID')}",
        "target_model_path": "./target_net_sp_bid.pth",
        "replay_buffer_path": "./replay_buffer_sp_bid.pkl",
        "temp_buffer_path": "./temp_buffer_sp_bid.pkl",
    },
    "SP_TOS": {
        "policy_model_path": f"./{running_model_name('SP_TOS')}",
        "target_model_path": "./target_net_sp_tos.pth",
        "replay_buffer_path": "./replay_buffer_sp_tos.pkl",
        "temp_buffer_path": "./temp_buffer_sp_tos.pkl",
    },
    "SP_PP": {
        "policy_model_path": f"./{running_model_name('SP_PP')}",
        "target_model_path": "./target_net_sp_pp.pth",
        "replay_buffer_path": "./replay_buffer_sp_pp.pkl",
        "temp_buffer_path": "./temp_buffer_sp_pp.pkl",
    },
    "SP_ROS": {
        "policy_model_path": f"./{running_model_name('SP_ROS')}",
        "target_model_path": "./target_net_sp_ros.pth",
        "replay_buffer_path": "./replay_buffer_sp_ros.pkl",
        "temp_buffer_path": "./temp_buffer_sp_ros.pkl",
    },
    "SPT_TOS": {
        "policy_model_path": f"./{running_model_name('SPT_TOS')}",
        "target_model_path": "./target_net_spt_tos.pth",
        "replay_buffer_path": "./replay_buffer_spt_tos.pkl",
        "temp_buffer_path": "./temp_buffer_spt_tos.pkl",
    },
    "SPT_PP": {
        "policy_model_path": f"./{running_model_name('SPT_PP')}",
        "target_model_path": "./target_net_spt_pp.pth",
        "replay_buffer_path": "./replay_buffer_spt_pp.pkl",
        "temp_buffer_path": "./temp_buffer_spt_pp.pkl",
    },
    "SPT_ROS": {
        "policy_model_path": f"./{running_model_name('SPT_ROS')}",
        "target_model_path": "./target_net_spt_ros.pth",
        "replay_buffer_path": "./replay_buffer_spt_ros.pkl",
        "temp_buffer_path": "./temp_buffer_spt_ros.pkl",
    },
    "SV_PP": {
        "policy_model_path": f"./{running_model_name('SV_PP')}",
        "target_model_path": "./target_net_sv_pp.pth",
        "replay_buffer_path": "./replay_buffer_sv_pp.pkl",
        "temp_buffer_path": "./temp_buffer_sv_pp.pkl",
    },
    "SV_TOS": {
        "policy_model_path": f"./{running_model_name('SV_TOS')}",
        "target_model_path": "./target_net_sv_tos.pth",
        "replay_buffer_path": "./replay_buffer_sv_tos.pkl",
        "temp_buffer_path": "./temp_buffer_sv_tos.pkl",
    },
    "SV_ROS": {
        "policy_model_path": f"./{running_model_name('SV_ROS')}",
        "target_model_path": "./target_net_sv_ros.pth",
        "replay_buffer_path": "./replay_buffer_sv_ros.pkl",
        "temp_buffer_path": "./temp_buffer_sv_ros.pkl",
    },
    "SV": {
        "policy_model_path": f"./{running_model_name('SV')}",
        "target_model_path": "./target_net_sv.pth",
        "replay_buffer_path": "./replay_buffer_sv.pkl",
        "temp_buffer_path": "./temp_buffer_sv.pkl",
    },
    "SD": {
        "policy_model_path": f"./{running_model_name('SD')}",
        "target_model_path": "./target_net_sd.pth",
        "replay_buffer_path": "./replay_buffer_sd.pkl",
        "temp_buffer_path": "./temp_buffer_sd.pkl",
    },
    "global": {
        "policy_model_path": f"./{running_model_name('global')}",
        "target_model_path": "./target_net_global.pth",
        "replay_buffer_path": "./replay_buffer_global.pkl",
        "temp_buffer_path": "./temp_buffer_global.pkl",
    },
}

scaler_ctr = StandardScaler()
scaler_cvr = StandardScaler()

scaler_ctr.fit(
    np.array(
        [
            -0.976,
            0,
            -0.944,
            -0.5,
            -0.875,
            0,
            -0.938,
            0,
            0.077,
            0,
            0,
            0,
            -0.667,
            0,
            0,
            0.004,
            0,
            0,
            0,
            -0.967,
            0,
            0,
            0.01,
            0,
            -0.97,
            -0.941,
            0,
            -0.5,
            -0.944,
            0,
            -0.75,
            -0.952,
            -0.5,
            -0.9,
            0.013,
            -0.973,
            0,
            -0.667,
            0.222,
            -0.971,
            0,
            -0.994,
            -0.917,
            0,
            -0.971,
            0,
            -0.5,
            -0.833,
            -0.95,
            -0.977,
            -0.955,
            -0.75,
            -0.987,
            0.143,
        ]
    ).reshape(-1, 1)
)
scaler_cvr.fit(
    np.array(
        [
            0,
            -0.5,
            0.0,
            -0.667,
            -0.909,
            -0.8,
            -0.5,
            -0.75,
            0.0,
            -0.857,
            0.25,
            0.00,
            -0.5,
            0.00,
            0.00,
            0.00,
            0.00,
            -0.5,
            -0.5,
            -0.5,
            -0.75,
            0.0,
            -0.9,
            0.0,
            0.0,
            0.0,
            -0.5,
            0.0,
        ]
    ).reshape(-1, 1)
)


def scaling_ctr(val: float):
    scaled_val = scaler_ctr.transform([[val]]).tolist()[0][0]
    return scaled_val


def scaling_cvr(val: float):
    scaled_val = scaler_cvr.transform([[val]]).tolist()[0][0]
    return scaled_val


def bid_from_action(min_value, step_size, action):
    return round(min_value + step_size * action, 2)


def action_from_bid(min_value: float, step_size: float, bid: float):
    return max(round((bid - min_value) / step_size), 0)


def negative_mapping(numer, denom):
    if denom <= 0:
        return 0
    elif numer <= 0:
        return (1 / denom) - 1
    return numer / denom


def decide_bucket(value, lower_limit, upper_limit, bucket_count):
    if bucket_count <= 0:
        raise ZeroDivisionError("Bucket count can not be equal or less than zero")

    bucket_size = (upper_limit - lower_limit) / bucket_count

    if value <= lower_limit:
        return 0
    if value >= upper_limit:
        return bucket_count - 1

    return int((value - lower_limit) / bucket_size)


def add_distributions(*distributions):
    return sum([np.array(dist) for dist in distributions]).tolist()


# ===>> Sponsored Videos
def get_videos_bids(keywordIds):
    key_list = sb.sponsored_brands_keyword_list(keywordIdList=keywordIds)
    return {str(k["keywordId"]): k["bid"] for k in key_list}


def get_enabled_video_campaigns(campaignIds):
    camp_list = sb.sponsored_brands_campaign_list(campaignIdList=campaignIds)
    return {
        c["campaignId"]: c.get("budget", 1)
        for c in camp_list
        if c["state"] == "ENABLED"
    }


def deep_q_videos(interval: int, monitor: bool):

    # Fetching data from SQL
    sql_data = qcm.fetch(
        f"""
DECLARE @NOW DATETIME = '{pst_time_str()}',
	@INTERVAL INT = {interval},
	@MONITOR BIT = {int(monitor)};
DECLARE @STARTOFHOUR DATETIME2 = CAST(FORMAT(@NOW, 'yyyy-MM-dd HH:00') AS DATETIME2);
DECLARE @CVRDAYCOUNT INT = (SELECT MAX(CVRDAYCOUNT) FROM MODELING_DB.DBO.Q_MASTER
    WHERE DEEPQSWITCH = 1 AND CAMPAIGNTYPE = 'SV');
DECLARE @TOSONLY BIT = (SELECT MAX(CAST(TOSONLY AS INT)) FROM MODELING_DB.DBO.Q_MASTER WHERE CAMPAIGNTYPE = 'SV');

WITH MAIN_TAB AS (
        SELECT * FROM MODELING_DB.DBO.Q_MASTER
        WHERE (DEEPQSWITCH = 1 OR TOSONLY = 1) AND CAMPAIGNTYPE = 'SV'),
KEY_TAB AS (
        SELECT * FROM MODELING_DB.DBO.KEYWORDS_FOR_AGENT
        WHERE CAMPAIGNID IN (SELECT CAMPAIGNID FROM MAIN_TAB)),
SQS_TAB AS (
	SELECT CAMPAIGN_ID, KEYWORD_ID, CLICKS, IMPRESSIONS, COST, DATATIME
	FROM [AMAZON_ANALYTICS].[DBO].AS_SB_TRAFFIC
	WHERE CAST(TIME_WINDOW_START AS DATE) = CAST(@NOW AS DATE)
		AND DATATIME <= @NOW AND CAMPAIGN_ID IN (SELECT CAMPAIGNID FROM MAIN_TAB)
        AND (@TOSONLY = 0 OR PLACEMENT_TYPE = 'TOP OF SEARCH ON-AMAZON')
	),
CURR_TAB AS (
	SELECT KEYWORD_ID, SUM(CLICKS) CLK0, SUM(IMPRESSIONS) IMP0, SUM(CAST(COST AS FLOAT)) COST0,
		CASE WHEN SUM(CLICKS) = 0 THEN 0
		ELSE CAST(SUM(COST) AS FLOAT)/SUM(CLICKS) END AS CPC,
        CASE WHEN SUM(IMPRESSIONS) = 0 THEN 0
            WHEN SUM(IMPRESSIONS) > 0 AND SUM(CLICKS) <= 0 THEN (1.0/SUM(IMPRESSIONS) - 1)
        ELSE CAST(SUM(CLICKS) AS FLOAT)/SUM(IMPRESSIONS) END AS CTR0
        FROM SQS_TAB GROUP BY KEYWORD_ID),
PREV_TAB AS (
        SELECT KEYWORD_ID, SUM(CLICKS) CLK1, SUM(IMPRESSIONS) IMP1,
		CASE WHEN SUM(CLICKS) = 0 THEN 0
		ELSE CAST(SUM(COST) AS FLOAT)/SUM(CLICKS) END AS CPC1,
            CASE WHEN SUM(IMPRESSIONS) = 0 THEN 0 WHEN SUM(IMPRESSIONS) > 0 AND SUM(CLICKS) <= 0 THEN (1.0/SUM(IMPRESSIONS) - 1)
        ELSE CAST(SUM(CLICKS) AS FLOAT)/SUM(IMPRESSIONS) END AS CTR1
        FROM SQS_TAB WHERE DATATIME <= DATEADD(MINUTE, -@INTERVAL, @NOW)
        GROUP BY KEYWORD_ID),
TOTAL_CLICK_TAB AS (
		SELECT KEYWORD_ID, SUM(CLICKS) AS CLICKS FROM [AMAZON_ANALYTICS].[DBO].AS_SB_TRAFFIC
		WHERE CAST(TIME_WINDOW_START AS DATETIME2) >= DATEADD(HOUR, -@CVRDAYCOUNT, @NOW) AND DATATIME <= @NOW
			AND CAMPAIGN_ID IN (SELECT CAMPAIGNID FROM MAIN_TAB)
		GROUP BY KEYWORD_ID),
TOTAL_CONV_TAB AS (
		SELECT KEYWORD_ID, SUM(ATTRIBUTED_CONVERSIONS_14D) AS CONVERSION FROM [AMAZON_ANALYTICS].[DBO].AS_SB_CONVERSION
		WHERE CAST(TIME_WINDOW_START AS DATETIME2) >= DATEADD(HOUR, -@CVRDAYCOUNT, @NOW) AND DATATIME <= @NOW
			AND CAMPAIGN_ID IN (SELECT CAMPAIGNID FROM MAIN_TAB)
		GROUP BY KEYWORD_ID),
CONV_TAB AS (
		SELECT SUM(CONVERSION) AS UNITS
		FROM (
			SELECT ISNULL(ORDERCOUNT - LAG(ORDERCOUNT) OVER (PARTITION BY CAST(TIMESTAMP AS DATE) ORDER BY TIMESTAMP), 0) AS CONVERSION
			FROM [AMAZON_MARKETING].[DBO].[AMZN_SALES_DATA]
			WHERE TIMESTAMP >= DATEADD(HOUR, -1, @STARTOFHOUR)
			  AND TIMESTAMP <= @NOW
		) AS TAB),
LATEST_CPC_TAB AS (
	SELECT KEYWORD_ID, MAX(DATATIME) LATEST
	FROM SQS_TAB WHERE COST > 0 GROUP BY KEYWORD_ID),
CPC_CURR_TAB AS(
	SELECT ST.KEYWORD_ID, SUM(CAST(ST.COST AS FLOAT)) / SUM(ST.CLICKS) AS CPC FROM
	SQS_TAB ST JOIN LATEST_CPC_TAB LCT ON ST.KEYWORD_ID = LCT.KEYWORD_ID AND ST.DATATIME = LCT.LATEST AND ST.CLICKS > 0
	GROUP BY ST.KEYWORD_ID
	),
CPC_PREV_TAB AS(
	SELECT ST.KEYWORD_ID, SUM(CAST(ST.COST AS FLOAT)) / SUM(ST.CLICKS) AS CPC FROM
	SQS_TAB ST JOIN(
		SELECT ST.KEYWORD_ID, MAX(ST.DATATIME) PREV
		FROM SQS_TAB ST JOIN LATEST_CPC_TAB LCT
			ON ST.KEYWORD_ID=LCT.KEYWORD_ID AND ST.DATATIME < LCT.LATEST AND ST.CLICKS > 0
		GROUP BY ST.KEYWORD_ID
	) PCT ON ST.KEYWORD_ID = PCT.KEYWORD_ID AND ST.DATATIME = PCT.PREV AND ST.CLICKS > 0
	GROUP BY ST.KEYWORD_ID),
COST_TAB AS (
	SELECT CAMPAIGN_ID, SUM(CAST(COST AS FLOAT)) COST FROM SQS_TAB GROUP BY CAMPAIGN_ID),
KEY_CONV_TAB AS (
	SELECT KEYWORD_ID, SUM(ATTRIBUTED_CONVERSIONS_14D) AS CONV FROM [AMAZON_ANALYTICS].[DBO].AS_SB_CONVERSION
		WHERE CAST(TIME_WINDOW_START AS DATE) >= CAST(@NOW AS DATE) AND DATATIME <= @NOW
			AND CAMPAIGN_ID IN (SELECT CAMPAIGNID FROM MAIN_TAB)
	GROUP BY KEYWORD_ID),
PREV_KEY_CONV_TAB AS (
	SELECT KEYWORD_ID, SUM(ATTRIBUTED_CONVERSIONS_14D) AS PREV_CONV FROM [AMAZON_ANALYTICS].[DBO].AS_SB_CONVERSION
		WHERE CAST(TIME_WINDOW_START AS DATE) >= CAST(@NOW AS DATE) AND DATATIME <=  DATEADD(MINUTE, -@INTERVAL, @NOW)
			AND CAMPAIGN_ID IN (SELECT CAMPAIGNID FROM MAIN_TAB)
	GROUP BY KEYWORD_ID),
FINAL_TAB AS (
	SELECT KT.KEYWORDID, KT.CAMPAIGNID, KT.ADGROUPID, MT.GAMMA, MT.LR, MT.EPSTART, MT.BASEBIDLOWER, MT.BASEBIDUPPER, MT.COMBINEDDEEPQSWITCH,
		MT.CVRFLAG, MT.CLICKSTEP, CASE WHEN TCV.CONVERSION>0 THEN CAST (TCLK.CLICKS AS FLOAT) / TCV.CONVERSION ELSE 0 END CVR,
        MT.CTRWEIGHT, MT.CONVWEIGHT, MT.CVRWEIGHT, MT.WEIGHTFLAG, MT.CPCWEIGHT, MT.BUDGETWEIGHT, MT.DYNAMICEXPLORE,
        MT.BOLTZFLAG, MT.TDECAY,
		ISNULL(ROUND(CT.CTR0,4),0) AS CURR_CTR, ROUND(ISNULL(CT.CTR0,0) - ISNULL(PT.CTR1,0),4) AS DELTA_CTR,
		ISNULL(ROUND(CT.CPC,2),0) AS CPC, ISNULL(ROUND(PT.CPC1,2),0) AS CPC1, ISNULL(CVT.UNITS,0) UNITS,
        ISNULL(CCT.CPC,0) CURR_CPC, ISNULL(CPT.CPC,0) PREV_CPC, ISNULL(CST.COST,0) SPENTBUDGET, ISNULL(KCT.CONV,0) CONV,
        ISNULL(PKCT.PREV_CONV,0) PREV_CONV, ISNULL(PT.CLK1,0) PREV_CLICK, ISNULL(CT.CLK0,0) CLICK, ISNULL(CT.COST0,0) COST
	FROM KEY_TAB KT LEFT JOIN CURR_TAB CT ON KT.KEYWORDID = CT.KEYWORD_ID
		LEFT JOIN MAIN_TAB MT ON KT.CAMPAIGNID = MT.CAMPAIGNID
		LEFT JOIN PREV_TAB PT ON KT.KEYWORDID = PT.KEYWORD_ID
		LEFT JOIN TOTAL_CLICK_TAB TCLK ON KT.KEYWORDID = TCLK.KEYWORD_ID
		LEFT JOIN TOTAL_CONV_TAB TCV ON KT.KEYWORDID = TCV.KEYWORD_ID
        LEFT JOIN CPC_CURR_TAB CCT ON KT.KEYWORDID = CCT.KEYWORD_ID
		LEFT JOIN CPC_PREV_TAB CPT ON KT.KEYWORDID = CPT.KEYWORD_ID
		LEFT JOIN COST_TAB CST ON KT.CAMPAIGNID = CST.CAMPAIGN_ID
		LEFT JOIN KEY_CONV_TAB KCT ON KT.KEYWORDID = KCT.KEYWORD_ID
		LEFT JOIN PREV_KEY_CONV_TAB PKCT ON KT.KEYWORDID = PKCT.KEYWORD_ID
        LEFT JOIN CONV_TAB CVT ON 1=1)

SELECT * FROM FINAL_TAB
WHERE @MONITOR = 0 OR DELTA_CTR <= 0 --OR CPC <> CPC1
"""
    )
    reward_distribution_list = [[0, 0, 0] for _ in range(100)]
    if sql_data == []:
        with open("sv_deepq.txt", "a") as f:
            f.write(pst_time_str() + "\n" + "No Data!" + "\n\n")
        return None, reward_distribution_list

    dynamic_explore = sql_data[0]["DYNAMICEXPLORE"]
    learning_rate = sql_data[0]["LR"]
    gamma = sql_data[0]["GAMMA"]
    epsilon = (
        sql_data[0]["EPSTART"]
        if not dynamic_explore
        else round(
            (min(0.99, sql_data[0]["EPSTART"]) ** (minute_from_last_reset / 60)), 2
        )
    )
    units = sql_data[0]["UNITS"]
    boltz_switch = sql_data[0]["BOLTZFLAG"]
    t_decay = sql_data[0]["TDECAY"]

    # Configuring the paths
    replay_buffer_path = path_dict["SV"]["replay_buffer_path"]
    temp_buffer_path = path_dict["SV"]["temp_buffer_path"]
    policy_model_path = path_dict["SV"]["policy_model_path"]
    target_model_path = path_dict["SV"]["target_model_path"]

    global_replay_buffer_path = path_dict["global"]["replay_buffer_path"]
    global_temp_buffer_path = path_dict["global"]["temp_buffer_path"]
    global_policy_model_path = path_dict["global"]["policy_model_path"]
    global_target_model_path = path_dict["global"]["target_model_path"]

    # Getting current bids from Amazon
    bid_dict = get_videos_bids([k["KEYWORDID"] for k in sql_data])
    # Getting currently enabled campaignIds
    enabled_campaignids = get_enabled_video_campaigns(
        list({c["CAMPAIGNID"] for c in sql_data})
    )

    # ===>> Readying the data for updating pending buffers
    pending_buffer_update_data = {
        r["KEYWORDID"]: {
            "next_state": torch.tensor(
                [
                    r["CURR_CTR"],
                    r["DELTA_CTR"],
                    r["CPC"],
                    bid_dict.get(r["KEYWORDID"], 0),
                    r["PREV_CPC"] - r["CURR_CPC"],
                    enabled_campaignids.get(r["CAMPAIGNID"], 1) - r["SPENTBUDGET"],
                ],
                dtype=torch.float32,
            ),
            "reward": r["CTRWEIGHT"] * r["DELTA_CTR"]
            + (
                r["BUDGETWEIGHT"]
                * (
                    negative_mapping(r["CONV"], r["CLICK"])
                    - negative_mapping(r["PREV_CONV"], r["PREV_CLICK"])
                )
            ),
        }
        for r in sql_data
        if bid_dict.get(r["KEYWORDID"], 0) >= 0.3
        and r["CAMPAIGNID"] in enabled_campaignids
    }
    global_pending_buffer_update_data = {}

    # Creating temporary buffer instant & updating the pending values
    temp_buffer = PersistentReplayBuffer(temp_buffer_path)
    temp_buffer.update(replay_buffer_path, pending_buffer_update_data)

    if policy_model_path == "./":
        dimensions = [6, 64, 64, 64, 64, 64, 64, 79]
        policy_model_path += "starting_model_sv_" + config_time_for_file_name() + ".pth"
    else:
        dimensions = get_dimensions_from_file("dqn_models", policy_model_path)

    if global_policy_model_path == "./":
        global_dimensions = [6, 64, 64, 64, 64, 64, 64, 79]
        global_policy_model_path += (
            "starting_model_global_" + config_time_for_file_name() + ".pth"
        )
    else:
        global_dimensions = get_dimensions_from_file(
            "dqn_models", global_policy_model_path
        )

    # ===>> Creating DQN agent
    dqn_agent = DQNAgent(
        dimensions=dimensions,
        replay_path=replay_buffer_path,
        temp_replay_path=temp_buffer_path,
        policy_net_path=policy_model_path,
        target_net_path=target_model_path,
        epsilon_start=epsilon,
        gamma=gamma,
        lr=learning_rate,
        campaignType="SV",
        boltz_switch=boltz_switch,
        t_decay=t_decay,
        t_start=1,
        t_end=0.1,
    )

    global_dqn_agent = DQNAgent(
        dimensions=global_dimensions,
        replay_path=global_replay_buffer_path,
        temp_replay_path=global_temp_buffer_path,
        policy_net_path=global_policy_model_path,
        target_net_path=global_target_model_path,
        lr=learning_rate,
        epsilon_start=epsilon,
        gamma=gamma,
        boltz_switch=boltz_switch,
        t_decay=t_decay,
        t_start=1,
        t_end=0.1,
    )
    loss_log = []

    # Training the network for 5 iterations
    for _ in range(5):
        loss_1 = dqn_agent.train_step()
        loss_2 = global_dqn_agent.train_step()
        if loss_1 is not None:
            loss_log.append({"agent": "SV", "loss": loss_1})
        if loss_2 is not None:
            loss_log.append({"agent": "GLOBAL", "loss": loss_2})
    # Saving models
    dqn_agent.save_models()
    global_dqn_agent.save_models()

    # ===>> Taking actions
    bid_update_list = []
    temp_buffer_list = []
    log = []

    for row in sql_data:
        keywordId = row["KEYWORDID"]
        adGroupId = row["ADGROUPID"]
        campaignId = row["CAMPAIGNID"]
        bid = bid_dict.get(keywordId, 0)
        cvr_condition = row["CVRFLAG"]
        global_deep_q = row["COMBINEDDEEPQSWITCH"]
        weight_flag = row["WEIGHTFLAG"]
        del_cpc = row["PREV_CPC"] - row["CURR_CPC"]
        budget = enabled_campaignids.get(campaignId, 1)

        max_cvr = 100
        if row["CVR"] <= row["CLICKSTEP"]:
            normalized_cvr = 0
        elif row["CVR"] >= 100:
            normalized_cvr = 1
        else:
            normalized_cvr = (row["CVR"] - row["CLICKSTEP"]) / (
                max_cvr - row["CLICKSTEP"]
            )

        if bid_dict.get(keywordId, 0) >= 0.3 and campaignId in enabled_campaignids:
            del_cvr = negative_mapping(row["CONV"], row["CLICK"]) - negative_mapping(
                row["PREV_CONV"], row["PREV_CLICK"]
            )
            reward = (
                (row["CTRWEIGHT"] * row["DELTA_CTR"])
                + (row["CONVWEIGHT"] * units / 10)
                + (
                    row["CPCWEIGHT"]
                    * (del_cpc / row["PREV_CPC"] if row["PREV_CPC"] > 0 else 0)
                )
                + (row["CVRWEIGHT"] * negative_mapping(row["CONV"], row["CLICK"]))
                # + (row["BUDGETWEIGHT"] * (budget - row["COST"]) / budget)
                + (row["BUDGETWEIGHT"] * del_cvr)
                if weight_flag
                else (row["DELTA_CTR"] + units / 10)
            )

            reward_distribution_list[decide_bucket(reward, -1, 1, 100)][0] += 1
            reward_distribution_list[decide_bucket(row["DELTA_CTR"], -1, 1, 100)][
                1
            ] += 1
            reward_distribution_list[decide_bucket(del_cvr, -1, 1, 100)][2] += 1

            global_pending_buffer_update_data.update(
                {
                    keywordId: {
                        "next_state": torch.tensor(
                            [
                                row["CURR_CTR"],
                                row["DELTA_CTR"],
                                row["CPC"],
                                bid_dict.get(keywordId, 0),
                                del_cpc,
                                budget - row["SPENTBUDGET"],
                            ],
                            dtype=torch.float32,
                        ),
                        "reward": reward,
                    }
                }
            )

        # Only taking actions if bid >= 0.3
        if bid >= 0.3 and campaignId in enabled_campaignids:
            action = (
                dqn_agent.select_action(
                    [
                        row["CURR_CTR"],
                        row["DELTA_CTR"],
                        row["CPC"],
                        bid,
                        del_cpc,
                        budget - row["SPENTBUDGET"],
                    ]
                )
                if not global_deep_q
                else global_dqn_agent.select_action(
                    [
                        row["CURR_CTR"],
                        row["DELTA_CTR"],
                        row["CPC"],
                        bid,
                        del_cpc,
                        budget - row["SPENTBUDGET"],
                    ]
                )
            )
            calculated_bid = bid_from_action(0.3, 0.2, action)

            # Sick joke of bounding the bid
            capped_bid = round(
                min(max(calculated_bid, row["BASEBIDLOWER"]), row["BASEBIDUPPER"]), 2
            )
            capped_action = action_from_bid(0.3, 0.2, capped_bid)

            if cvr_condition:
                if random.uniform(0, 1) > normalized_cvr:
                    app_bid = capped_bid
                else:
                    app_bid = 0.3
                    capped_action = 0
                    log.append(
                        {
                            "keywordId": keywordId,
                            "cvrn": True,
                            "prevBid": bid,
                        }
                    )

            else:
                app_bid = capped_bid

            # For updating bid in Amazon
            bid_update_list.append(
                {
                    "campaignId": campaignId,
                    "adGroupId": adGroupId,
                    "keywordId": keywordId,
                    "bid": app_bid,
                }
            )
            # Creating buffer for new buffer acquisition in Temporary Buffer file
            temp_buffer_list.append(
                {
                    "unique_id": keywordId,
                    "state": torch.tensor(
                        [
                            row["CURR_CTR"],
                            row["DELTA_CTR"],
                            row["CPC"],
                            bid,
                            del_cpc,
                            budget - row["SPENTBUDGET"],
                        ],
                        dtype=torch.float32,
                    ),
                    "action": capped_action,
                    "next_state": None,
                    "reward": None,
                    "done": False,
                }
            )

    global_temp_buffer = PersistentReplayBuffer(global_temp_buffer_path)
    global_temp_buffer.update(
        global_replay_buffer_path, global_pending_buffer_update_data
    )

    if bid_update_list:
        sb.UPDATE_KEYWORDS_SB(bid_update_list)

    if temp_buffer_list:
        temp_buffer.store(temp_buffer_list)
        global_temp_buffer.store(temp_buffer_list)

    batch_rewards = [v["reward"] for v in global_pending_buffer_update_data.values()]
    avg_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else None

    individual_loss = [item["loss"] for item in loss_log if item["agent"] == "SV"]
    avg_sv_loss = sum(individual_loss) / len(individual_loss) if individual_loss else 0

    global_loss = [item["loss"] for item in loss_log if item["agent"] == "GLOBAL"]
    avg_global_loss = sum(global_loss) / len(global_loss) if global_loss else 0

    loss_dict = {"Individual": avg_sv_loss, "Global": avg_global_loss}

    with open("sv_deepq.txt", "a") as f:
        f.write(
            pst_time_str()
            + "\n"
            + str(bid_update_list)
            + "\n"
            + str(log)
            + "\n"
            + str(loss_dict)
            + "\n\n"
        )

    return avg_reward, reward_distribution_list


# ===>> Sponsored Dispay
def get_display_bids(keywordIds):
    key_list = sd.sponsored_display_target_list(
        targetIdList=keywordIds, states=["enabled"]
    )
    return {str(k["targetId"]): k.get("bid", 0) for k in key_list}


def get_enabled_display_campaigns(campaignIds):
    camp_list = sd.sponsored_display_campaigns_list(campaignIdList=campaignIds)
    return {
        str(c["campaignId"]): c.get("budget", 1)
        for c in camp_list
        if c["state"] == "enabled"
    }


def deep_q_display(interval: int, monitor: bool):

    # Fetching data from SQL
    sql_data = qcm.fetch(
        f"""
DECLARE @NOW DATETIME = '{pst_time_str()}',
	@INTERVAL INT = {interval},
	@MONITOR BIT = {int(monitor)};
DECLARE @STARTOFHOUR DATETIME2 = CAST(FORMAT(@NOW, 'yyyy-MM-dd HH:00') AS DATETIME2);
DECLARE @CVRDAYCOUNT INT = (SELECT MAX(CVRDAYCOUNT) FROM MODELING_DB.DBO.Q_MASTER
    WHERE DEEPQSWITCH = 1 AND CAMPAIGNTYPE = 'SD');

WITH MAIN_TAB AS (
        SELECT * FROM MODELING_DB.DBO.Q_MASTER
        WHERE DEEPQSWITCH = 1 AND CAMPAIGNTYPE = 'SD'),
KEY_TAB AS (
        SELECT * FROM MODELING_DB.DBO.KEYWORDS_FOR_AGENT
        WHERE CAMPAIGNID IN (SELECT CAMPAIGNID FROM MAIN_TAB)),
SQS_TAB AS (
	SELECT CAMPAIGN_ID, TARGET_ID, CLICKS, IMPRESSIONS, COST, DATATIME
	FROM [AMAZON_ANALYTICS].[DBO].AS_SD_TRAFFIC
	WHERE CAST(TIME_WINDOW_START AS DATE) = CAST(@NOW AS DATE)
		AND DATATIME <= @NOW AND CAMPAIGN_ID IN (SELECT CAMPAIGNID FROM MAIN_TAB)
	),
CURR_TAB AS (
	SELECT TARGET_ID, SUM(CLICKS) CLK0, SUM(IMPRESSIONS) IMP0,  SUM(CAST(COST AS FLOAT)) COST0,
		CASE WHEN SUM(CLICKS) = 0 THEN 0
		ELSE CAST(SUM(COST) AS FLOAT)/SUM(CLICKS) END AS CPC,
        CASE WHEN SUM(IMPRESSIONS) = 0 THEN 0
            WHEN SUM(IMPRESSIONS) > 0 AND SUM(CLICKS) <= 0 THEN (1.0/SUM(IMPRESSIONS) - 1)
        ELSE CAST(SUM(CLICKS) AS FLOAT)/SUM(IMPRESSIONS) END AS CTR0
        FROM SQS_TAB GROUP BY TARGET_ID),
PREV_TAB AS (
        SELECT TARGET_ID, SUM(CLICKS) CLK1, SUM(IMPRESSIONS) IMP1,
		CASE WHEN SUM(CLICKS) = 0 THEN 0
		ELSE CAST(SUM(COST) AS FLOAT)/SUM(CLICKS) END AS CPC1,
            CASE WHEN SUM(IMPRESSIONS) = 0 THEN 0 WHEN SUM(IMPRESSIONS) > 0 AND SUM(CLICKS) <= 0 THEN (1.0/SUM(IMPRESSIONS) - 1)
        ELSE CAST(SUM(CLICKS) AS FLOAT)/SUM(IMPRESSIONS) END AS CTR1
        FROM SQS_TAB WHERE DATATIME <= DATEADD(MINUTE, -@INTERVAL, @NOW)
        GROUP BY TARGET_ID),
TOTAL_CLICK_TAB AS (
		SELECT TARGET_ID, SUM(CLICKS) AS CLICKS FROM [AMAZON_ANALYTICS].[DBO].AS_SD_TRAFFIC
		WHERE CAST(TIME_WINDOW_START AS DATETIME2) >= DATEADD(HOUR, -@CVRDAYCOUNT, @NOW) AND CAST(DATATIME AS DATE) <= @NOW
			AND CAMPAIGN_ID IN (SELECT CAMPAIGNID FROM MAIN_TAB)
		GROUP BY TARGET_ID),
TOTAL_CONV_TAB AS (
		SELECT TARGET_ID, SUM(ATTRIBUTED_CONVERSIONS_14D) AS CONVERSION FROM [AMAZON_ANALYTICS].[DBO].AS_SD_CONVERSION
		WHERE CAST(TIME_WINDOW_START AS DATETIME2) >= DATEADD(HOUR, -@CVRDAYCOUNT, @NOW) AND CAST(DATATIME AS DATE) <= @NOW
			AND CAMPAIGN_ID IN (SELECT CAMPAIGNID FROM MAIN_TAB)
		GROUP BY TARGET_ID),
CONV_TAB AS (
		SELECT SUM(CONVERSION) AS UNITS
		FROM (
			SELECT ISNULL(ORDERCOUNT - LAG(ORDERCOUNT) OVER (PARTITION BY CAST(TIMESTAMP AS DATE) ORDER BY TIMESTAMP), 0) AS CONVERSION
			FROM [AMAZON_MARKETING].[DBO].[AMZN_SALES_DATA]
			WHERE TIMESTAMP >= DATEADD(HOUR, -1, @STARTOFHOUR)
			  AND TIMESTAMP <= @NOW
		) AS TAB),
LATEST_CPC_TAB AS (
	SELECT TARGET_ID, MAX(DATATIME) LATEST
	FROM SQS_TAB WHERE COST > 0 GROUP BY TARGET_ID),
CPC_CURR_TAB AS(
	SELECT ST.TARGET_ID, SUM(CAST(ST.COST AS FLOAT)) / SUM(ST.CLICKS) AS CPC FROM
	SQS_TAB ST JOIN LATEST_CPC_TAB LCT ON ST.TARGET_ID = LCT.TARGET_ID AND ST.DATATIME = LCT.LATEST AND ST.CLICKS > 0
	GROUP BY ST.TARGET_ID
	),
CPC_PREV_TAB AS(
	SELECT ST.TARGET_ID, SUM(CAST(ST.COST AS FLOAT)) / SUM(ST.CLICKS) AS CPC FROM
	SQS_TAB ST JOIN(
		SELECT ST.TARGET_ID, MAX(ST.DATATIME) PREV
		FROM SQS_TAB ST JOIN LATEST_CPC_TAB LCT
			ON ST.TARGET_ID=LCT.TARGET_ID AND ST.DATATIME < LCT.LATEST AND ST.COST > 0 AND ST.CLICKS > 0
		GROUP BY ST.TARGET_ID
	) PCT ON ST.TARGET_ID = PCT.TARGET_ID AND ST.DATATIME = PCT.PREV
	GROUP BY ST.TARGET_ID),
COST_TAB AS (
	SELECT CAMPAIGN_ID, SUM(CAST(COST AS FLOAT)) COST FROM SQS_TAB GROUP BY CAMPAIGN_ID),
KEY_CONV_TAB AS (
	SELECT TARGET_ID AS KEYWORDID , SUM(ATTRIBUTED_CONVERSIONS_14D) AS CONV FROM [AMAZON_ANALYTICS].[DBO].AS_SD_CONVERSION
		WHERE CAST(TIME_WINDOW_START AS DATE) >= CAST(@NOW AS DATE) AND DATATIME <= @NOW
			AND CAMPAIGN_ID IN (SELECT CAMPAIGNID FROM MAIN_TAB)
	GROUP BY TARGET_ID),
PREV_KEY_CONV_TAB AS (
	SELECT TARGET_ID AS KEYWORDID , SUM(ATTRIBUTED_CONVERSIONS_14D) AS PREV_CONV FROM [AMAZON_ANALYTICS].[DBO].AS_SD_CONVERSION
		WHERE CAST(TIME_WINDOW_START AS DATE) >= CAST(@NOW AS DATE) AND DATATIME <=  DATEADD(MINUTE, -@INTERVAL, @NOW)
			AND CAMPAIGN_ID IN (SELECT CAMPAIGNID FROM MAIN_TAB)
	GROUP BY TARGET_ID),
FINAL_TAB AS (
	SELECT KT.KEYWORDID, KT.CAMPAIGNID, KT.ADGROUPID, MT.GAMMA, MT.LR, MT.EPSTART, MT.BASEBIDLOWER, MT.BASEBIDUPPER, MT.COMBINEDDEEPQSWITCH,
		MT.CVRFLAG, MT.CLICKSTEP, CASE WHEN TCV.CONVERSION>0 THEN CAST (TCLK.CLICKS AS FLOAT) / TCV.CONVERSION ELSE 0 END CVR,
        MT.CTRWEIGHT, MT.CONVWEIGHT, MT.CVRWEIGHT, MT.WEIGHTFLAG, MT.CPCWEIGHT, MT.BUDGETWEIGHT, MT.DYNAMICEXPLORE,
        MT.BOLTZFLAG, MT.TDECAY,
		ISNULL(ROUND(CT.CTR0,4),0) AS CURR_CTR,
		ROUND(ISNULL(CT.CTR0, 0) - ISNULL(PT.CTR1, 0), 4) AS DELTA_CTR,
		ISNULL(ROUND(CT.CPC,2),0) AS CPC, ISNULL(ROUND(PT.CPC1,2),0) AS CPC1, ISNULL(CVT.UNITS,0) UNITS,
        ISNULL(CCT.CPC, 0) CURR_CPC, ISNULL(CPT.CPC, 0) PREV_CPC,  ISNULL(CST.COST,0) SPENTBUDGET, ISNULL(KCT.CONV,0) CONV,
		ISNULL(PKCT.PREV_CONV,0) PREV_CONV, ISNULL(PT.CLK1,0) PREV_CLICK, ISNULL(CT.CLK0,0) CLICK, ISNULL(CT.COST0,0) COST
	FROM KEY_TAB KT LEFT JOIN CURR_TAB CT ON KT.KEYWORDID = CT.TARGET_ID
		LEFT JOIN MAIN_TAB MT ON KT.CAMPAIGNID = MT.CAMPAIGNID
		LEFT JOIN PREV_TAB PT ON KT.KEYWORDID = PT.TARGET_ID
		LEFT JOIN TOTAL_CLICK_TAB TCLK ON KT.KEYWORDID = TCLK.TARGET_ID
		LEFT JOIN TOTAL_CONV_TAB TCV ON KT.KEYWORDID = TCV.TARGET_ID
        LEFT JOIN CPC_CURR_TAB CCT ON KT.KEYWORDID = CCT.TARGET_ID
		LEFT JOIN CPC_PREV_TAB CPT ON KT.KEYWORDID = CPT.TARGET_ID
		LEFT JOIN COST_TAB CST ON KT.CAMPAIGNID = CST.CAMPAIGN_ID
		LEFT JOIN KEY_CONV_TAB KCT ON KT.KEYWORDID = KCT.KEYWORDID
        LEFT JOIN PREV_KEY_CONV_TAB PKCT ON KT.KEYWORDID = PKCT.KEYWORDID
        LEFT JOIN CONV_TAB CVT ON 1=1)
SELECT * FROM FINAL_TAB
WHERE @MONITOR = 0 OR DELTA_CTR <= 0 --OR CPC <> CPC1
"""
    )
    reward_distribution_list = [[0, 0, 0] for _ in range(100)]
    if sql_data == []:
        with open("sd_deepq.txt", "a") as f:
            f.write(pst_time_str() + "\n" + "No Data!" + "\n\n")
        return None, reward_distribution_list

    dynamic_explore = sql_data[0]["DYNAMICEXPLORE"]
    learning_rate = sql_data[0]["LR"]
    gamma = sql_data[0]["GAMMA"]
    epsilon = (
        sql_data[0]["EPSTART"]
        if not dynamic_explore
        else round(
            (min(0.99, sql_data[0]["EPSTART"]) ** (minute_from_last_reset / 60)), 2
        )
    )
    units = sql_data[0]["UNITS"]
    boltz_switch = sql_data[0]["BOLTZFLAG"]
    t_decay = sql_data[0]["TDECAY"]

    # Configuring the paths
    replay_buffer_path = path_dict["SD"]["replay_buffer_path"]
    temp_buffer_path = path_dict["SD"]["temp_buffer_path"]
    policy_model_path = path_dict["SD"]["policy_model_path"]
    target_model_path = path_dict["SD"]["target_model_path"]

    global_replay_buffer_path = path_dict["global"]["replay_buffer_path"]
    global_temp_buffer_path = path_dict["global"]["temp_buffer_path"]
    global_policy_model_path = path_dict["global"]["policy_model_path"]
    global_target_model_path = path_dict["global"]["target_model_path"]

    # Getting current bids from Amazon
    bid_dict = get_display_bids([k["KEYWORDID"] for k in sql_data])
    # Getting currently enabled campaignIds
    enabled_campaignids = get_enabled_display_campaigns(
        list({c["CAMPAIGNID"] for c in sql_data})
    )

    # ===>> Readying the data for updating pending buffers
    pending_buffer_update_data = {
        r["KEYWORDID"]: {
            "next_state": torch.tensor(
                [
                    r["CURR_CTR"],
                    r["DELTA_CTR"],
                    r["CPC"],
                    bid_dict.get(r["KEYWORDID"], 0),
                    r["PREV_CPC"] - r["CURR_CPC"],
                    enabled_campaignids.get(r["CAMPAIGNID"], 1) - r["SPENTBUDGET"],
                ],
                dtype=torch.float32,
            ),
            "reward": r["CTRWEIGHT"] * r["DELTA_CTR"]
            + (
                r["BUDGETWEIGHT"]
                * (
                    negative_mapping(r["CONV"], r["CLICK"])
                    - negative_mapping(r["PREV_CONV"], r["PREV_CLICK"])
                )
            ),
        }
        for r in sql_data
        if bid_dict.get(r["KEYWORDID"], 0) >= 0.3
        and r["CAMPAIGNID"] in enabled_campaignids
    }
    global_pending_buffer_update_data = {}

    # Creating temporary buffer instant & updating the pending values
    temp_buffer = PersistentReplayBuffer(temp_buffer_path)
    temp_buffer.update(replay_buffer_path, pending_buffer_update_data)

    if policy_model_path == "./":
        dimensions = [6, 64, 64, 64, 64, 64, 64, 79]
        policy_model_path += "starting_model_sd_" + config_time_for_file_name() + ".pth"
    else:
        dimensions = get_dimensions_from_file("dqn_models", policy_model_path)

    if global_policy_model_path == "./":
        global_dimensions = [6, 64, 64, 64, 64, 64, 64, 79]
        global_policy_model_path += (
            "starting_model_global_" + config_time_for_file_name() + ".pth"
        )
    else:
        global_dimensions = get_dimensions_from_file(
            "dqn_models", global_policy_model_path
        )

    # ===>> Creating DQN agent
    dqn_agent = DQNAgent(
        dimensions=dimensions,
        replay_path=replay_buffer_path,
        temp_replay_path=temp_buffer_path,
        policy_net_path=policy_model_path,
        target_net_path=target_model_path,
        epsilon_start=epsilon,
        gamma=gamma,
        lr=learning_rate,
        campaignType="SD",
        boltz_switch=boltz_switch,
        t_decay=t_decay,
        t_start=1,
        t_end=0.1,
    )

    global_dqn_agent = DQNAgent(
        dimensions=global_dimensions,
        replay_path=global_replay_buffer_path,
        temp_replay_path=global_temp_buffer_path,
        policy_net_path=global_policy_model_path,
        target_net_path=global_target_model_path,
        lr=learning_rate,
        epsilon_start=epsilon,
        gamma=gamma,
        boltz_switch=boltz_switch,
        t_decay=t_decay,
        t_start=1,
        t_end=0.1,
    )
    loss_log = []

    # Training the network for 5 iterations
    for _ in range(5):
        loss_1 = dqn_agent.train_step()
        loss_2 = global_dqn_agent.train_step()
        if loss_1 is not None:
            loss_log.append({"agent": "SD", "loss": loss_1})
        if loss_2 is not None:
            loss_log.append({"agent": "GLOBAL", "loss": loss_2})
    # Saving models
    dqn_agent.save_models()
    global_dqn_agent.save_models()

    # ===>> Taking actions
    bid_update_list = []
    temp_buffer_list = []
    log = []

    for row in sql_data:
        keywordId = row["KEYWORDID"]
        adGroupId = row["ADGROUPID"]
        campaignId = row["CAMPAIGNID"]
        bid = bid_dict.get(keywordId, 0)
        cvr_condition = row["CVRFLAG"]
        global_deep_q = row["COMBINEDDEEPQSWITCH"]
        weight_flag = row["WEIGHTFLAG"]
        del_cpc = row["PREV_CPC"] - row["CURR_CPC"]
        budget = enabled_campaignids.get(campaignId, 1)

        max_cvr = 100
        if row["CVR"] <= row["CLICKSTEP"]:
            normalized_cvr = 0
        elif row["CVR"] >= 100:
            normalized_cvr = 1
        else:
            normalized_cvr = (row["CVR"] - row["CLICKSTEP"]) / (
                max_cvr - row["CLICKSTEP"]
            )

        if bid >= 0.3 and campaignId in enabled_campaignids:
            del_cvr = negative_mapping(row["CONV"], row["CLICK"]) - negative_mapping(
                row["PREV_CONV"], row["PREV_CLICK"]
            )
            reward = (
                (row["CTRWEIGHT"] * row["DELTA_CTR"])
                + (row["CONVWEIGHT"] * units / 10)
                + (
                    row["CPCWEIGHT"]
                    * (del_cpc / row["PREV_CPC"] if row["PREV_CPC"] > 0 else 0)
                )
                + (row["CVRWEIGHT"] * negative_mapping(row["CONV"], row["CLICK"]))
                # + (row["BUDGETWEIGHT"] * (budget - row["COST"]) / budget)
                + (row["BUDGETWEIGHT"] * (del_cvr))
                if weight_flag
                else row["DELTA_CTR"] + units / 10
            )

            reward_distribution_list[decide_bucket(reward, -1, 1, 100)][0] += 1
            reward_distribution_list[decide_bucket(row["DELTA_CTR"], -1, 1, 100)][
                1
            ] += 1
            reward_distribution_list[decide_bucket(del_cvr, -1, 1, 100)][2] += 1

            global_pending_buffer_update_data.update(
                {
                    keywordId: {
                        "next_state": torch.tensor(
                            [
                                row["CURR_CTR"],
                                row["DELTA_CTR"],
                                row["CPC"],
                                bid,
                                del_cpc,
                                budget - row["SPENTBUDGET"],
                            ],
                            dtype=torch.float32,
                        ),
                        "reward": reward,
                    }
                }
            )

        # Only taking actions if bid >= 0.3
        if bid >= 0.3 and campaignId in enabled_campaignids:
            action = (
                dqn_agent.select_action(
                    [
                        row["CURR_CTR"],
                        row["DELTA_CTR"],
                        row["CPC"],
                        bid,
                        del_cpc,
                        budget - row["SPENTBUDGET"],
                    ]
                )
                if not global_deep_q
                else global_dqn_agent.select_action(
                    [
                        row["CURR_CTR"],
                        row["DELTA_CTR"],
                        row["CPC"],
                        bid,
                        del_cpc,
                        budget - row["SPENTBUDGET"],
                    ]
                )
            )
            calculated_bid = bid_from_action(0.3, 0.2, action)

            # Sick joke of bounding the bid
            capped_bid = round(
                min(max(calculated_bid, row["BASEBIDLOWER"]), row["BASEBIDUPPER"]), 2
            )
            capped_action = action_from_bid(0.3, 0.2, capped_bid)

            if cvr_condition:
                if random.uniform(0, 1) > normalized_cvr:
                    app_bid = capped_bid
                else:
                    app_bid = 0.3
                    capped_action = 0
                    log.append(
                        {
                            "keywordId": keywordId,
                            "cvrn": True,
                            "prevBid": bid,
                        }
                    )

            else:
                app_bid = capped_bid

            # For updating bid in Amazon
            bid_update_list.append(
                {
                    "campaignId": campaignId,
                    "adGroupId": adGroupId,
                    "targetId": keywordId,
                    "bid": app_bid,
                }
            )
            # Creating buffer for new buffer acquisition in Temporary Buffer file
            temp_buffer_list.append(
                {
                    "unique_id": keywordId,
                    "state": torch.tensor(
                        [
                            row["CURR_CTR"],
                            row["DELTA_CTR"],
                            row["CPC"],
                            bid,
                            del_cpc,
                            budget - row["SPENTBUDGET"],
                        ],
                        dtype=torch.float32,
                    ),
                    "action": capped_action,
                    "next_state": None,
                    "reward": None,
                    "done": False,
                }
            )

    global_temp_buffer = PersistentReplayBuffer(global_temp_buffer_path)
    global_temp_buffer.update(
        global_replay_buffer_path, global_pending_buffer_update_data
    )

    if bid_update_list:
        sd.UPDATE_KEYWORDS_SD(bid_update_list)

    if temp_buffer_list:
        temp_buffer.store(temp_buffer_list)
        global_temp_buffer.store(temp_buffer_list)

    batch_rewards = [v["reward"] for v in global_pending_buffer_update_data.values()]
    avg_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else None

    individual_loss = [item["loss"] for item in loss_log if item["agent"] == "SD"]
    avg_sv_loss = sum(individual_loss) / len(individual_loss) if individual_loss else 0

    global_loss = [item["loss"] for item in loss_log if item["agent"] == "GLOBAL"]
    avg_global_loss = sum(global_loss) / len(global_loss) if global_loss else 0

    loss_log = {"Individual": avg_sv_loss, "Global": avg_global_loss}

    with open("sd_deepq.txt", "a") as f:
        f.write(
            pst_time_str()
            + "\n"
            + str(bid_update_list)
            + "\n"
            + str(log)
            + "\n"
            + str(loss_log)
            + "\n\n"
        )

    return avg_reward, reward_distribution_list


# ===> Sponsored Products & PAT
def get_boost_from_totalbid(base_bid: float, total_bid: float):
    return round(max(min((total_bid / base_bid - 1) * 100, 899), 3))


def get_totalbid_from_boost(base_bid: float, boost: int):
    return base_bid * (1 + boost / 100)


def get_sp_spt_campaign_detail(campaignIds):
    camp_list = sp.sponsored_products_campaign_list(campaignIdList=campaignIds)
    boost_dict = {}

    for camp in camp_list:
        if camp["state"] == "ENABLED":
            tos, pp, ros = 0, 0, 0
            for place in camp.get("dynamicBidding", {}).get("placementBidding", []):
                if place["placement"] == "PLACEMENT_REST_OF_SEARCH":
                    ros = place["percentage"]
                if place["placement"] == "PLACEMENT_PRODUCT_PAGE":
                    pp = place["percentage"]
                if place["placement"] == "PLACEMENT_TOP":
                    tos = place["percentage"]

            boost_dict[camp["campaignId"]] = {
                "TOS": tos,
                "PP": pp,
                "ROS": ros,
                "strategy": camp.get("dynamicBidding", {}).get(
                    "strategy", "LEGACY_FOR_SALES"
                ),
                "budget": camp.get("budget", {}).get("budget", 1),
            }

    return boost_dict


def get_sv_campaign_detail(campaignIds):
    camp_list = sb.sponsored_brands_campaign_list(campaignIdList=campaignIds)
    boost_dict = {}

    for camp in camp_list:
        if camp["state"] == "ENABLED":
            pp, tos, ros = 0, 0, 0
            for place in camp.get("bidding", {}).get("bidAdjustmentsByPlacement", []):
                if place["placement"] == "OTHER":
                    ros = place["percentage"]
                if place["placement"] == "TOP_OF_SEARCH":
                    tos = place["percentage"]
                if place["placement"] == "DETAIL_PAGE":
                    pp = place["percentage"]

            boost_dict[camp["campaignId"]] = {
                "PP": pp,
                "TOS": tos,
                "ROS": ros,
                "strategy": camp.get("bidding", {}).get(
                    "bidOptimizationStrategy", "MAXIMIZE_IMMEDIATE_SALES"
                ),
                "budget": camp.get("budget", 1),
            }

    return boost_dict


def get_products_bids(campaignIds):
    key_list = sp.sponsored_products_keyword_list(campaignIdList=campaignIds)
    bid_dict = {}

    for key in key_list:
        if key.get("bid", 0) >= 0.03:
            if key["campaignId"] not in bid_dict:
                bid_dict[key["campaignId"]] = [key["bid"]]
            else:
                bid_dict[key["campaignId"]] += [key["bid"]]

    return {k: max(v) for k, v in bid_dict.items()}


def get_products_bids_byKey(keywordIds):
    key_list = sp.sponsored_products_keyword_list(keywordIdList=keywordIds)

    return {k["keywordId"]: k.get("bid", 0) for k in key_list}


def get_pat_bids(campaignIds):
    target_list = sp.sponsored_products_target_list(campaignIdList=campaignIds)
    bid_dict = {}

    for target in target_list:
        if target.get("bid", 0) >= 0.03:
            if target["campaignId"] not in bid_dict:
                bid_dict[target["campaignId"]] = [target["bid"]]
            else:
                bid_dict[target["campaignId"]] += [target["bid"]]

    return {k: max(v) for k, v in bid_dict.items()}


def get_video_avgbids(campaignIds):
    key_list = sb.sponsored_brands_keyword_list(
        campaignIdList=campaignIds, states=["enabled"]
    )
    bid_dict = {}

    for key in key_list:
        if key.get("bid", 0) >= 0.3:
            if key["campaignId"] not in bid_dict:
                bid_dict[key["campaignId"]] = [key["bid"]]
            else:
                bid_dict[key["campaignId"]] += [key["bid"]]

    return {str(k): sum(v) / len(v) for k, v in bid_dict.items()}


def create_json_for_boost_update(data: dict[str, dict]):
    boost_update_list = []
    for campId, detail in data.items():
        placementBidList = []
        if "TOS" in detail:
            placementBidList.append(
                {"placement": "PLACEMENT_TOP", "percentage": detail["TOS"]}
            )
        if "PP" in detail:
            placementBidList.append(
                {"placement": "PLACEMENT_PRODUCT_PAGE", "percentage": detail["PP"]}
            )
        if "ROS" in detail:
            placementBidList.append(
                {"placement": "PLACEMENT_REST_OF_SEARCH", "percentage": detail["ROS"]}
            )

        boost_update_list.append(
            {
                "campaignId": campId,
                "dynamicBidding": {
                    "placementBidding": placementBidList,
                    "strategy": detail["strategy"],
                },
            }
        )
    return boost_update_list


def create_json_for_boost_update_sv(data: dict[str, dict]):
    boost_update_list = []
    for campId, detail in data.items():
        placementBidList = []
        if "PP" in detail:
            placementBidList.append(
                {"placement": "DETAIL_PAGE", "percentage": detail["PP"]}
            )
        if "TOS" in detail:
            placementBidList.append(
                {"placement": "TOP_OF_SEARCH", "percentage": detail["TOS"]}
            )
        if "ROS" in detail:
            placementBidList.append({"placement": "OTHER", "percentage": detail["ROS"]})

        boost_update_list.append(
            {
                "campaignId": campId,
                "bidding": {
                    "bidOptimization": False,
                    "bidAdjustmentsByPlacement": placementBidList,
                    "bidOptimizationStrategy": detail["strategy"],
                },
            }
        )
    return boost_update_list

def scale_imp(impression_value):
    pst = pst_date()  # midnight PST today

    result = qcm.fetch("""
        WITH daily_keyword_stats AS (
            SELECT
                keyword_id,
                CAST(LEFT(time_window_start, 10) AS DATE) AS day,
                SUM(impressions) AS daily_impression,
                SUM(clicks) AS daily_clicks
            FROM AMAZON_MARKETING.DBO.as_sp_traffic
            WHERE CAST(LEFT(time_window_start, 10) AS DATE)
                  >= DATEADD(DAY, -30, ?)
              GROUP BY
                keyword_id,
                CAST(LEFT(time_window_start, 10) AS DATE)
            HAVING
                SUM(impressions) > 0
                AND SUM(clicks) > 0
        ),
        keyword_stats AS (
            SELECT
                keyword_id,
                MIN(daily_impression) AS min_daily_impression,
                MAX(daily_impression) AS max_daily_impression
            FROM daily_keyword_stats
            GROUP BY keyword_id
        )
        SELECT
            AVG(min_daily_impression) AS min_impression,
            AVG(max_daily_impression) AS max_impression
        FROM keyword_stats;
    """, [pst])[0]

    min_imp = result["min_impression"]
    max_imp = result["max_impression"]

    if min_imp == max_imp:
        return 0.0

    return (impression_value - min_imp) / (max_imp - min_imp)






# using this function for sp, spt and sv boost changes
def run_deep_q_for_sp_spt(
    data: list[dict],
    campaignType: str,
    placement: str,
    learning_rate: float,
    gamma: float,
    epsilon: float,
    boltz_switch: bool,
    t_decay: float,
    total_imp: int = None,
):
    # ===>> Readying the data for updating pending buffers
    replay_buffer_path = path_dict[campaignType + "_" + placement]["replay_buffer_path"]
    temp_buffer_path = path_dict[campaignType + "_" + placement]["temp_buffer_path"]
    policy_model_path = path_dict[campaignType + "_" + placement]["policy_model_path"]
    target_model_path = path_dict[campaignType + "_" + placement]["target_model_path"]

    global_replay_buffer_path = path_dict["global"]["replay_buffer_path"]
    global_temp_buffer_path = path_dict["global"]["temp_buffer_path"]
    global_policy_model_path = path_dict["global"]["policy_model_path"]
    global_target_model_path = path_dict["global"]["target_model_path"]

    random_number = (
        np.random.uniform(-0.01, 0.1)
        if pst_time() > pst_date() + timedelta(hours=2)
        else -1
    )
    to_store_buffer = True if pst_time().minute <= 3 else False

    pending_buffer_update_data = {
        r["CAMPAIGNID"]: {
            "next_state": torch.tensor(
                [
                    r["CURR_CTR"],
                    r["DELTA_CTR"],
                    r["CPC"],
                    r["TOTALBID"],
                    (13.377 - r["SHORT_IMP"]) / 25.565,
                    (13.377 - r["LONG_IMP"]) / 25.565,
                ],
                dtype=torch.float32,
            ),
            "reward": r["CTRWEIGHT"] * r["DELTA_CTR"]
            + (
                r["BUDGETWEIGHT"]
                * (
                    negative_mapping(r["CONV"], r["CLICK"])
                    - negative_mapping(r["PREV_CONV"], r["PREV_CLICK"])
                )
            ),
        }
        for r in data
        if r["BOOST"] >= 3
        and (
            (total_imp is None)
            or (r.get("ALL_PLC_IMP", 0) / max(total_imp, 1) >= random_number)
        )
    }
    reward_distribution_list = [[0, 0, 0] for _ in range(100)]
    if not pending_buffer_update_data:
        return {}, [], None, reward_distribution_list, {}

    global_pending_buffer_update_data = {}

    # Creating temporary buffer instant & updating the pending values
    temp_buffer = PersistentReplayBuffer(temp_buffer_path)
    if to_store_buffer:
        temp_buffer.update(replay_buffer_path, pending_buffer_update_data)

    if policy_model_path == "./":
        dimensions = [6, 64, 64, 64, 64, 64, 64, 79]
        policy_model_path += (
            "starting_model_"
            + f"{campaignType.lower()}_{placement.lower()}_"
            + config_time_for_file_name()
            + ".pth"
        )
    else:
        dimensions = get_dimensions_from_file("dqn_models", policy_model_path)

    if global_policy_model_path == "./":
        global_dimensions = [6, 64, 64, 64, 64, 64, 64, 79]
        global_policy_model_path += (
            "starting_model_global_" + config_time_for_file_name() + ".pth"
        )
    else:
        global_dimensions = get_dimensions_from_file(
            "dqn_models", global_policy_model_path
        )

    # ===>> Creating DQN agent
    dqn_agent = DQNAgent(
        dimensions=dimensions,
        replay_path=replay_buffer_path,
        temp_replay_path=temp_buffer_path,
        policy_net_path=policy_model_path,
        target_net_path=target_model_path,
        epsilon_start=epsilon,
        gamma=gamma,
        lr=learning_rate,
        campaignType=campaignType,
        placement=placement,
        boltz_switch=boltz_switch,
        t_decay=t_decay,
        t_start=1,
        t_end=0.1,
    )

    global_dqn_agent = DQNAgent(
        dimensions=global_dimensions,
        replay_path=global_replay_buffer_path,
        temp_replay_path=global_temp_buffer_path,
        policy_net_path=global_policy_model_path,
        target_net_path=global_target_model_path,
        lr=learning_rate,
        epsilon_start=epsilon,
        gamma=gamma,
        boltz_switch=boltz_switch,
        t_decay=t_decay,
        t_start=1,
        t_end=0.1,
    )
    loss_log = []

    # Training the network for 100 iterations
    for _ in range(100):
        loss_1 = dqn_agent.train_step()
        loss_2 = global_dqn_agent.train_step()
        if loss_1 is not None:
            loss_log.append({"agent": "Ind", "loss": loss_1})
        if loss_2 is not None:
            loss_log.append({"agent": "GLOBAL", "loss": loss_2})
    # Saving models
    dqn_agent.save_models()
    global_dqn_agent.save_models()

    boost_update_dict = {}
    temp_buffer_list = []
    log = []

    for row in data:
        allowed = (total_imp is None) or (
            row.get("ALL_PLC_IMP", 0) / max(total_imp, 1) >= random_number
        )
        cvr_condition = row["CVRFLAG"]
        global_deep_q = row["COMBINEDDEEPQSWITCH"]
        weight_flag = row["WEIGHTFLAG"]
        delta_ctr = row["DELTA_CTR"]
        only_drop = row.get("ONLYDROP", False)
        drop_when_improved = row.get("DROPIMPROVED", False)

        max_cvr = 100
        if row["CVR"] <= row["CLICK_STEP"]:
            normalized_cvr = 0
        elif row["CVR"] >= 100:
            normalized_cvr = 1
        else:
            normalized_cvr = (row["CVR"] - row["CLICK_STEP"]) / (
                max_cvr - row["CLICK_STEP"]
            )

        if row["BOOST"] >= 3 and allowed:
            del_cvr = scaling_cvr(
                negative_mapping(row["CONV"], row["CLICK"])
                - negative_mapping(row["PREV_CONV"], row["PREV_CLICK"])
            )
            reward = (
                (row["CTRWEIGHT"] * scaling_ctr(row["DELTA_CTR"]))
                # + (row["CONVWEIGHT"] * row.get("CTR_SHR", 0) * row["UNITS"])
                - (row["CPCWEIGHT"] * (row["COST"]))
                + (
                    row["CVRWEIGHT"]
                    * scaling_cvr(negative_mapping(row["CONV"], row["CLICK"]))
                )
                # + ( row["BUDGETWEIGHT"] * (row["BUDGET"] - row["COST"]) / row["BUDGET"])
                + (row["BUDGETWEIGHT"] * del_cvr) - scale_imp(row["IMP_INC"])
                if weight_flag
                else (row["DELTA_CTR"] + row["UNITS"] / 10)
            )

            reward_distribution_list[decide_bucket(reward, -1, 1, 100)][0] += 1
            reward_distribution_list[decide_bucket(row["DELTA_CTR"], -1, 1, 100)][
                1
            ] += 1
            reward_distribution_list[decide_bucket(del_cvr, -1, 1, 100)][2] += 1

            global_pending_buffer_update_data[row["CAMPAIGNID"]] = {
                "next_state": torch.tensor(
                    [
                        row["CURR_CTR"],
                        row["DELTA_CTR"],
                        row["CPC"],
                        row["TOTALBID"],
                        (13.377 - row["SHORT_IMP"]) / 25.565,
                        (13.377 - row["LONG_IMP"]) / 25.565,
                    ],
                    dtype=torch.float32,
                ),
                "reward": reward,
            }

        if row["BOOST"] >= 3 and allowed:
            action = (
                dqn_agent.select_action(
                    [
                        row["CURR_CTR"],
                        row["DELTA_CTR"],
                        row["CPC"],
                        row["TOTALBID"],
                        (13.377 - row["SHORT_IMP"]) / 25.565,
                        (13.377 - row["LONG_IMP"]) / 25.565,
                    ]
                )
                if not global_deep_q
                else global_dqn_agent.select_action(
                    [
                        row["CURR_CTR"],
                        row["DELTA_CTR"],
                        row["CPC"],
                        row["TOTALBID"],
                        (13.377 - row["SHORT_IMP"]) / 25.565,
                        (13.377 - row["LONG_IMP"]) / 25.565,
                    ]
                )
            )
            calculated_bid = bid_from_action(0.3, 0.2, action)
            capped_bid = max(min(calculated_bid, row["BIDUPPER"]), row["BIDLOWER"])
            capped_action = action_from_bid(0.3, 0.2, capped_bid)
            calculated_boost = get_boost_from_totalbid(row["BASEBID"], capped_bid)

            if cvr_condition:
                if random.uniform(0, 1) > normalized_cvr:
                    app_boost = calculated_boost
                else:
                    app_boost = 3
                    capped_action = 0
                    log.append(
                        {
                            "campaignId": row["CAMPAIGNID"],
                            "cvrn": True,
                            "prevBoost": row["BOOST"],
                        }
                    )

            else:
                app_boost = calculated_boost

            if only_drop and delta_ctr >= 0:
                app_boost = row["BOOST"]
            if drop_when_improved and delta_ctr > 0:
                app_boost = 6

            if app_boost != row["BOOST"]:
                boost_update_dict[row["CAMPAIGNID"]] = {
                    placement: app_boost,
                    "strategy": row["STRATEGY"],
                }

            temp_buffer_list.append(
                {
                    "unique_id": row["CAMPAIGNID"],
                    "state": torch.tensor(
                        [
                            row["CURR_CTR"],
                            row["DELTA_CTR"],
                            row["CPC"],
                            row["TOTALBID"],
                            (13.377 - row["SHORT_IMP"]) / 25.565,
                            (13.377 - row["LONG_IMP"]) / 25.565,
                        ],
                        dtype=torch.float32,
                    ),
                    "action": capped_action,
                    "next_state": None,
                    "reward": None,
                    "done": False,
                }
            )

    global_temp_buffer = PersistentReplayBuffer(global_temp_buffer_path)
    if to_store_buffer:
        global_temp_buffer.update(
            global_replay_buffer_path, global_pending_buffer_update_data
        )

    if temp_buffer_list and to_store_buffer:
        temp_buffer.store(temp_buffer_list)
        global_temp_buffer.store(temp_buffer_list)

    batch_rewards = [v["reward"] for v in global_pending_buffer_update_data.values()]
    avg_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else None

    individual_loss = [item["loss"] for item in loss_log if item["agent"] == "Ind"]
    avg_sv_loss = sum(individual_loss) / len(individual_loss) if individual_loss else 0

    global_loss = [item["loss"] for item in loss_log if item["agent"] == "GLOBAL"]
    avg_global_loss = sum(global_loss) / len(global_loss) if global_loss else 0

    loss_dict = {"Individual": avg_sv_loss, "Global": avg_global_loss}

    return boost_update_dict, log, avg_reward, reward_distribution_list, loss_dict


def deep_q_sp(interval: int, monitor: bool):
    sql_data = qcm.fetch(
        f"""
--FOR SP
DECLARE @NOW DATETIME = '{pst_time_str()}',
	@INTERVAL INT = {interval},
	@MONITOR BIT = {int(monitor)};
DECLARE @STARTOFHOUR DATETIME2 = CAST(FORMAT(@NOW, 'yyyy-MM-dd HH:00') AS DATETIME2);
DECLARE @CVRDAYCOUNT INT = (SELECT MAX(CVRDAYCOUNT) FROM MODELING_DB.DBO.Q_MASTER WITH (NOLOCK)
    WHERE (DEEPTOSSWITCH = 1 OR DEEPPPSWITCH = 1 OR DEEPROSSWITCH = 1) AND CAMPAIGNTYPE IN ('SP'));
DECLARE @IMPVOLUMEHOUR INT = (SELECT MAX(IMPVOLUMEHOUR) FROM MODELING_DB.DBO.Q_MASTER WITH (NOLOCK)
    WHERE (DEEPTOSSWITCH = 1 OR DEEPPPSWITCH = 1 OR DEEPROSSWITCH = 1) AND CAMPAIGNTYPE IN ('SP'));
DECLARE @SHORTIMPHOUR INT = (SELECT MAX(SHORTIMPHOUR) FROM MODELING_DB.DBO.Q_MASTER WITH (NOLOCK)
    WHERE (DEEPTOSSWITCH = 1 OR DEEPPPSWITCH = 1 OR DEEPROSSWITCH = 1) AND CAMPAIGNTYPE IN ('SP'));
DECLARE @SAMPLEHOUR INT = (SELECT MAX(SAMPLEHOUR) FROM MODELING_DB.DBO.Q_MASTER WITH (NOLOCK)
    WHERE (DEEPTOSSWITCH = 1 OR DEEPPPSWITCH = 1 OR DEEPROSSWITCH = 1) AND CAMPAIGNTYPE IN ('SP'));
DECLARE @ITERATIONCOUNT INT = (SELECT MAX(ITERATIONCOUNT) FROM MODELING_DB.DBO.Q_MASTER WITH (NOLOCK)
    WHERE (DEEPTOSSWITCH = 1 OR DEEPPPSWITCH = 1 OR DEEPROSSWITCH = 1) AND CAMPAIGNTYPE IN ('SP'));
DECLARE @IMPRESSIONTHRESHOLD INT = (SELECT MAX(IMPRESSIONTHRESHOLD) FROM MODELING_DB.DBO.Q_MASTER
    WHERE (DEEPTOSSWITCH = 1 OR DEEPPPSWITCH = 1 OR DEEPROSSWITCH = 1) AND CAMPAIGNTYPE IN ('SP'));
DECLARE @HOURLYTARGETIMP INT = (SELECT MAX(HOURLYTARGETIMP) FROM MODELING_DB.DBO.Q_MASTER WITH (NOLOCK)
    WHERE (DEEPTOSSWITCH = 1 OR DEEPPPSWITCH = 1 OR DEEPROSSWITCH = 1) AND CAMPAIGNTYPE IN ('SV'));

WITH MAIN_TAB AS (
    SELECT * FROM MODELING_DB.DBO.Q_MASTER WITH (NOLOCK)
    WHERE (DEEPTOSSWITCH = 1 OR DEEPPPSWITCH = 1 OR DEEPROSSWITCH = 1) AND CAMPAIGNTYPE IN ('SP')),

TIME_TAB AS (
	SELECT CAMPAIGN_ID, PLACEMENT, MAX([TIMESTAMP]) MAXTIME
	FROM [AMAZON_MARKETING].[DBO].AS_SP_TRAFFIC WITH (NOLOCK)
	WHERE CAST(TIME_WINDOW_START AS DATE) = CAST(@NOW AS DATE)
	AND [TIMESTAMP] <= @NOW AND CAMPAIGN_ID IN (SELECT CAMPAIGNID FROM MAIN_TAB) GROUP BY campaign_id, PLACEMENT),

SQS_TAB AS(
	SELECT CAMPAIGN_ID, PLACEMENT, CLICKS, IMPRESSIONS, COST, [TIMESTAMP]
	FROM [AMAZON_MARKETING].[DBO].AS_SP_TRAFFIC WITH (NOLOCK)
	WHERE CAST(TIME_WINDOW_START AS DATE) = CAST(@NOW AS DATE)
		 AND CAMPAIGN_ID IN (SELECT CAMPAIGNID FROM MAIN_TAB)),

SQS_TAB2 AS (
	SELECT A.CAMPAIGN_ID, A.PLACEMENT, A.CLICKS, A.IMPRESSIONS, A.COST, A.[TIMESTAMP],TIME_TAB.MAXTIME
	FROM [AMAZON_MARKETING].[DBO].AS_SP_TRAFFIC A WITH (NOLOCK)
	JOIN TIME_TAB ON A.CAMPAIGN_ID = TIME_TAB.CAMPAIGN_ID AND A.PLACEMENT = TIME_TAB.PLACEMENT
	WHERE CAST(TIME_WINDOW_START AS DATE) = CAST(@NOW AS DATE)
		AND [TIMESTAMP] <= @NOW AND A.CAMPAIGN_ID IN (SELECT CAMPAIGNID FROM MAIN_TAB)),

CURR_TAB AS (
    SELECT CAMPAIGN_ID, PLACEMENT, SUM(CLICKS) CLK0, SUM(IMPRESSIONS) IMP0, SUM(COST) COST0,
		CASE WHEN SUM(CLICKS) = 0 THEN 0 ELSE CAST(SUM(COST) AS FLOAT)/SUM(CLICKS) END AS CPC,
        CASE WHEN SUM(IMPRESSIONS) = 0 THEN 0 WHEN SUM(IMPRESSIONS) > 0 AND SUM(CLICKS) <= 0 THEN (1.0/SUM(IMPRESSIONS) - 1)
        ELSE CAST(SUM(CLICKS) AS FLOAT)/SUM(IMPRESSIONS) END AS CTR0
    FROM SQS_TAB GROUP BY CAMPAIGN_ID, PLACEMENT),
PREV_TAB AS (
    SELECT CAMPAIGN_ID, PLACEMENT, SUM(CLICKS) CLK1, SUM(IMPRESSIONS) IMP1,
		CASE WHEN SUM(CLICKS) = 0 THEN 0 ELSE CAST(SUM(COST) AS FLOAT)/SUM(CLICKS) END AS CPC1,
        CASE WHEN SUM(IMPRESSIONS) = 0 THEN 0 WHEN SUM(IMPRESSIONS) > 0 AND SUM(CLICKS) <= 0 THEN (1.0/SUM(IMPRESSIONS) - 1)
		ELSE CAST(SUM(CLICKS) AS FLOAT)/SUM(IMPRESSIONS) END AS CTR1
    FROM SQS_TAB WHERE [TIMESTAMP] <= DATEADD(MINUTE, -@INTERVAL, @NOW)
    GROUP BY CAMPAIGN_ID, PLACEMENT),
BAL_IMP_TAB AS(
	SELECT CAMPAIGN_ID, PLACEMENT, 100*(MAX(NOW_IMP) - MAX(BAL_ITER_IMP))/(MAX(BAL_ITER_IMP)+1) IMP_INC FROM(
		SELECT CAMPAIGN_ID, PLACEMENT, CASE WHEN @HOURLYTARGETIMP = -1 
				THEN CAST(SUM(IMPRESSIONS)AS FLOAT)/@ITERATIONCOUNT ELSE @HOURLYTARGETIMP END BAL_ITER_IMP, 0 NOW_IMP
		FROM SQS_TAB WHERE IMPRESSIONS >= 0 AND TIMESTAMP BETWEEN DATEADD(MINUTE, -60*@ITERATIONCOUNT, @NOW) AND DATEADD(MINUTE, -60, @NOW)
		GROUP BY CAMPAIGN_ID, PLACEMENT
		UNION
		SELECT CAMPAIGN_ID, PLACEMENT, 0 BAL_ITER_IMP, CAST(SUM(IMPRESSIONS)AS FLOAT) NOW_IMP
		FROM SQS_TAB2 WHERE IMPRESSIONS >= 0 AND TIMESTAMP BETWEEN DATEADD(MINUTE, -60, (CASE WHEN @HOURLYTARGETIMP = -1 THEN MAXTIME ELSE @NOW END)) 
		AND (CASE WHEN @HOURLYTARGETIMP = -1 THEN MAXTIME ELSE @NOW END)
		GROUP BY CAMPAIGN_ID, PLACEMENT)T
    GROUP BY CAMPAIGN_ID, PLACEMENT),
TOTAL_CLICK_TAB AS (
	SELECT CAMPAIGN_ID, PLACEMENT, SUM(CLICKS) AS CLICKS FROM [AMAZON_MARKETING].[DBO].AS_SP_TRAFFIC WITH (NOLOCK)
	WHERE CAST(TIME_WINDOW_START AS DATETIME2) >= DATEADD(HOUR, -@CVRDAYCOUNT, @NOW) AND CAST(TIMESTAMP AS DATE) <= @NOW
	GROUP BY CAMPAIGN_ID, PLACEMENT),
LONG_IMP_TAB AS (
	SELECT CAMPAIGN_ID, PLACEMENT, CAST(IMP AS FLOAT) LONG_IMP
	FROM (SELECT CAMPAIGN_ID, PLACEMENT, SUM(IMPRESSIONS) AS IMP FROM [AMAZON_MARKETING].[DBO].AS_SP_TRAFFIC WITH (NOLOCK)
		WHERE TIMESTAMP BETWEEN DATEADD(HOUR, -@IMPVOLUMEHOUR, @NOW) AND @NOW
		GROUP BY CAMPAIGN_ID, PLACEMENT)T),
SHORT_IMP_TAB AS (
	SELECT CAMPAIGN_ID, PLACEMENT, CAST(IMP AS FLOAT) SHORT_IMP
	FROM (SELECT CAMPAIGN_ID, PLACEMENT, SUM(IMPRESSIONS) AS IMP FROM [AMAZON_MARKETING].[DBO].AS_SP_TRAFFIC WITH (NOLOCK)
		WHERE TIMESTAMP BETWEEN DATEADD(HOUR, -@SHORTIMPHOUR, @NOW) AND @NOW
		GROUP BY CAMPAIGN_ID, PLACEMENT)T),
SAMPLE_IMP_TAB AS (
	SELECT CAMPAIGN_ID, PLACEMENT, SUM(IMPRESSIONS) AS SAMPLE_IMP FROM [AMAZON_MARKETING].[DBO].AS_SP_TRAFFIC WITH (NOLOCK)
		WHERE TIMESTAMP BETWEEN DATEADD(HOUR, -@SAMPLEHOUR, @NOW) AND @NOW AND CAST(TIME_WINDOW_START AS DATE) = CAST(@NOW AS DATE)
		GROUP BY CAMPAIGN_ID, PLACEMENT),
TOTAL_CONV_TAB AS (
	SELECT CAMPAIGN_ID, PLACEMENT, SUM(CONVERSIONS_14D) AS CONVERSION FROM [AMAZON_MARKETING].[DBO].AS_SP_CONVERSION WITH (NOLOCK)
	WHERE CAST(TIME_WINDOW_START AS DATETIME2) >= DATEADD(HOUR, -@CVRDAYCOUNT, @NOW) AND CAST(TIMESTAMP AS DATE) <= @NOW
	GROUP BY CAMPAIGN_ID, PLACEMENT),
CVR_TAB AS (
	SELECT TCLK.CAMPAIGN_ID, TCLK.PLACEMENT, TCLK.CLICKS, TCV.CONVERSION,
		CASE WHEN TCV.CONVERSION>0 THEN CAST (TCLK.CLICKS AS FLOAT) / TCV.CONVERSION ELSE 0 END CVR
	FROM TOTAL_CLICK_TAB TCLK LEFT JOIN TOTAL_CONV_TAB TCV
		ON TCLK.CAMPAIGN_ID = TCV.CAMPAIGN_ID AND TCLK.PLACEMENT = TCV.PLACEMENT),
LATEST_CPC_TAB AS (
	SELECT CAMPAIGN_ID, PLACEMENT, MAX([TIMESTAMP]) LATEST
	FROM SQS_TAB WHERE COST > 0 GROUP BY CAMPAIGN_ID, PLACEMENT),
CPC_CURR_TAB AS(
	SELECT ST.CAMPAIGN_ID, ST.PLACEMENT, SUM(CAST(ST.COST AS FLOAT)) / SUM(ST.CLICKS) AS CPC FROM
	SQS_TAB ST JOIN LATEST_CPC_TAB LCT ON ST.CAMPAIGN_ID = LCT.CAMPAIGN_ID AND ST.PLACEMENT = LCT.PLACEMENT AND ST.[TIMESTAMP] = LCT.LATEST AND ST.CLICKS > 0
	GROUP BY ST.CAMPAIGN_ID, ST.PLACEMENT),
CPC_PREV_TAB AS(
	SELECT ST.CAMPAIGN_ID, ST.PLACEMENT, SUM(CAST(ST.COST AS FLOAT)) / SUM(ST.CLICKS) AS CPC FROM
	SQS_TAB ST JOIN(
		SELECT ST.CAMPAIGN_ID, ST.PLACEMENT, MAX(ST.[TIMESTAMP]) PREV
		FROM SQS_TAB ST JOIN LATEST_CPC_TAB LCT
			ON ST.CAMPAIGN_ID=LCT.CAMPAIGN_ID AND ST.PLACEMENT=LCT.PLACEMENT AND ST.[TIMESTAMP] < LCT.LATEST AND ST.CLICKS > 0
		GROUP BY ST.CAMPAIGN_ID, ST.PLACEMENT
	) PCT ON ST.CAMPAIGN_ID = PCT.CAMPAIGN_ID AND ST.PLACEMENT = PCT.PLACEMENT AND ST.[TIMESTAMP] = PCT.PREV AND ST.CLICKS > 0
	GROUP BY ST.CAMPAIGN_ID, ST.PLACEMENT),
PLCMNT_CONV_TAB AS (
	SELECT CAMPAIGN_ID, PLACEMENT, SUM(CONVERSIONS_14D) AS CONV FROM [AMAZON_MARKETING].[DBO].AS_SP_CONVERSION WITH (NOLOCK)
	WHERE CAST(TIME_WINDOW_START AS DATE) = CAST(@NOW AS DATE) AND [TIMESTAMP] <= @NOW
		AND CAMPAIGN_ID IN (SELECT CAMPAIGNID FROM MAIN_TAB)
	GROUP BY CAMPAIGN_ID, PLACEMENT),
PREV_PLCMNT_CONV_TAB AS (
	SELECT CAMPAIGN_ID, PLACEMENT, SUM(CONVERSIONS_14D) AS PREV_CONV FROM [AMAZON_MARKETING].[DBO].AS_SP_CONVERSION WITH (NOLOCK)
	WHERE CAST(TIME_WINDOW_START AS DATE) = CAST(@NOW AS DATE) AND [TIMESTAMP] <= DATEADD(MINUTE, -@INTERVAL, @NOW)
		AND CAMPAIGN_ID IN (SELECT CAMPAIGNID FROM MAIN_TAB)
	GROUP BY CAMPAIGN_ID, PLACEMENT),
JOINED_TAB AS (
	SELECT CT.CAMPAIGN_ID, CT.PLACEMENT, CT.CLK0, CT.IMP0, CT.CTR0, CT.CPC, CAST(CT.COST0 AS FLOAT) COST0, LIT.LONG_IMP, SIT.SHORT_IMP,
		PT.CLK1, PT.IMP1, PT.CTR1, PT.CPC1, CVT.CVR, CPC_C.CPC AS CURR_CPC, CPC_P.CPC AS PREV_CPC, PCT.CONV, PVCT.PREV_CONV, SMIT.SAMPLE_IMP,
		CASE WHEN @HOURLYTARGETIMP != -1 THEN
				CASE WHEN ISNULL([BIT].IMP_INC, 0) < @IMPRESSIONTHRESHOLD THEN 0 ELSE ISNULL([BIT].IMP_INC, 0) END 
			ELSE ISNULL([BIT].IMP_INC, 0) END IMP_INC,
		CASE WHEN ISNULL(CT.CLK0, 0) - ISNULL(PT.CLK1, 0) <= 0 OR ISNULL(CT.IMP0, 0) - ISNULL(PT.IMP1, 0) <= 0 THEN 0
			ELSE CAST(ISNULL(CT.CLK0, 0) - ISNULL(PT.CLK1, 0) AS FLOAT) / (ISNULL(CT.IMP0, 0) - ISNULL(PT.IMP1, 0)) END CTR_IMPRV
	FROM CURR_TAB CT LEFT JOIN PREV_TAB PT ON CT.CAMPAIGN_ID=PT.CAMPAIGN_ID AND CT.PLACEMENT=PT.PLACEMENT
		LEFT JOIN CVR_TAB CVT ON CT.CAMPAIGN_ID=CVT.CAMPAIGN_ID AND CT.PLACEMENT=CVT.PLACEMENT
        LEFT JOIN CPC_CURR_TAB CPC_C ON CT.CAMPAIGN_ID=CPC_C.CAMPAIGN_ID AND CT.PLACEMENT=CPC_C.PLACEMENT
		LEFT JOIN CPC_PREV_TAB CPC_P ON CT.CAMPAIGN_ID=CPC_P.CAMPAIGN_ID AND CT.PLACEMENT=CPC_P.PLACEMENT
        LEFT JOIN PREV_PLCMNT_CONV_TAB PVCT ON CT.CAMPAIGN_ID=PVCT.CAMPAIGN_ID AND CT.PLACEMENT=PVCT.PLACEMENT
		LEFT JOIN LONG_IMP_TAB LIT ON CT.CAMPAIGN_ID=LIT.CAMPAIGN_ID AND CT.PLACEMENT=LIT.PLACEMENT
		LEFT JOIN SHORT_IMP_TAB SIT ON CT.CAMPAIGN_ID=SIT.CAMPAIGN_ID AND CT.PLACEMENT=SIT.PLACEMENT
		LEFT JOIN SAMPLE_IMP_TAB SMIT ON CT.CAMPAIGN_ID=SMIT.CAMPAIGN_ID AND CT.PLACEMENT=SMIT.PLACEMENT
		LEFT JOIN PLCMNT_CONV_TAB PCT ON CT.CAMPAIGN_ID=PCT.CAMPAIGN_ID AND CT.PLACEMENT=PCT.PLACEMENT
		LEFT JOIN BAL_IMP_TAB [BIT] ON CT.CAMPAIGN_ID=[BIT].CAMPAIGN_ID AND CT.PLACEMENT = [BIT].PLACEMENT),
JOINED_TAB1 AS (
	SELECT *, CASE WHEN SUM(CTR_IMPRV) OVER () <= 0 THEN 0 ELSE CTR_IMPRV / SUM(CTR_IMPRV) OVER () END CTR_SHARE
	FROM JOINED_TAB),
CONV_TAB AS (
		SELECT SUM(CONVERSION) AS UNITS
		FROM (
			SELECT ISNULL(ORDERCOUNT - LAG(ORDERCOUNT) OVER (PARTITION BY CAST(TIMESTAMP AS DATE) ORDER BY TIMESTAMP), 0) AS CONVERSION
			FROM [AMAZON_MARKETING].[DBO].[AMZN_SALES_DATA] WITH (NOLOCK)
			WHERE TIMESTAMP >= DATEADD(HOUR, -1, @STARTOFHOUR)
			  AND TIMESTAMP <= @NOW
		) AS TAB),
COST_TAB AS (
	SELECT CAMPAIGN_ID, SUM(CAST(COST AS FLOAT)) COST FROM SQS_TAB GROUP BY CAMPAIGN_ID),
MERGED_TAB AS(
	SELECT MT.CAMPAIGNID,MT.GAMMA, MT.LR, MT.EPSTART, MT.DEEPTOSSWITCH, MT.DEEPPPSWITCH, MT.DEEPROSSWITCH, MT.COMBINEDDEEPQSWITCH,
		MT.TOSBIDLOWER, MT.TOSBIDUPPER, MT.PPBIDLOWER, MT.PPBIDUPPER, MT.ROSBIDLOWER, MT.ROSBIDUPPER, MT.CVRFLAG, MT.DYNAMICEXPLORE,
		MT.PPCLICKSTEP, MT.ROSCLICKSTEP, MT.TOSCLICKSTEP, ISNULL(PP_T.CVR,0) PP_CVR, ISNULL(ROS_T.CVR,0) ROS_CVR, ISNULL(TOS_T.CVR,0) TOS_CVR,
        MT.WEIGHTFLAG, MT.CTRWEIGHT, MT.CONVWEIGHT, MT.CVRWEIGHT, MT.CPCWEIGHT, MT.BUDGETWEIGHT, MT.ONLYDROP, MT.DROPIMPROVED,
        MT.BOLTZFLAG, MT.TDECAY, MT.BUFFERFLAG,
		ISNULL(PP_T.SAMPLE_IMP,0) PP_IMP, ISNULL(ROS_T.SAMPLE_IMP, 0) ROS_IMP, ISNULL(TOS_T.SAMPLE_IMP, 0) TOS_IMP,
		ISNULL(PP_T.CTR0,0) PP_CURR_CTR, ISNULL(ROS_T.CTR0,0) ROS_CURR_CTR, ISNULL(TOS_T.CTR0,0) TOS_CURR_CTR,
		ISNULL(ROUND(TOS_T.CTR0 - ISNULL(TOS_T.CTR1,0),4),0) TOS_DELTA_CTR,
		ISNULL(ROUND(PP_T.CTR0 - ISNULL(PP_T.CTR1,0),4),0) PP_DELTA_CTR, ISNULL(ROUND(ROS_T.CTR0 - ISNULL(ROS_T.CTR1,0),4),0) ROS_DELTA_CTR,
		ISNULL(PP_T.CPC,0) PP_CPC, ISNULL(ROS_T.CPC,0) ROS_CPC, ISNULL(TOS_T.CPC,0) TOS_CPC,
		ISNULL(PP_T.CPC1,0) PP_CPC1, ISNULL(ROS_T.CPC1,0) ROS_CPC1, ISNULL(TOS_T.CPC1,0) TOS_CPC1,
        ISNULL(PP_T.CURR_CPC,0) PP_CURR_CPC, ISNULL(ROS_T.CURR_CPC,0) ROS_CURR_CPC, ISNULL(TOS_T.CURR_CPC,0) TOS_CURR_CPC,
		ISNULL(PP_T.PREV_CPC,0) PP_PREV_CPC, ISNULL(ROS_T.PREV_CPC,0) ROS_PREV_CPC, ISNULL(TOS_T.PREV_CPC,0) TOS_PREV_CPC,
		ISNULL(PP_T.CLK1,0) PP_PREV_CLK, ISNULL(TOS_T.CLK1,0) TOS_PREV_CLK, ISNULL(ROS_T.CLK1,0) ROS_PREV_CLK,
		ISNULL(PP_T.CLK0,0) PP_CLK, ISNULL(TOS_T.CLK0,0) TOS_CLK, ISNULL(ROS_T.CLK0,0) ROS_CLK,
		ISNULL(PP_T.COST0,0) PP_COST, ISNULL(TOS_T.COST0,0) TOS_COST, ISNULL(ROS_T.COST0,0) ROS_COST,
		ISNULL(PP_T.PREV_CONV,0) PP_PREV_CONV, ISNULL(TOS_T.PREV_CONV,0) TOS_PREV_CONV, ISNULL(ROS_T.PREV_CONV,0) ROS_PREV_CONV,
		ISNULL(PP_T.CONV,0) PP_CONV, ISNULL(TOS_T.CONV,0) TOS_CONV, ISNULL(ROS_T.CONV,0) ROS_CONV,
		ISNULL(PP_T.CTR_SHARE, 0) PP_CTR_SHR, ISNULL(TOS_T.CTR_SHARE, 0) TOS_CTR_SHR, ISNULL(ROS_T.CTR_SHARE, 0) ROS_CTR_SHR,
		ISNULL(PP_T.LONG_IMP, 0) PP_LONG_IMP,  ISNULL(TOS_T.LONG_IMP, 0) TOS_LONG_IMP, ISNULL(ROS_T.LONG_IMP, 0) ROS_LONG_IMP,
		ISNULL(PP_T.SHORT_IMP, 0) PP_SHORT_IMP,  ISNULL(TOS_T.SHORT_IMP, 0) TOS_SHORT_IMP, ISNULL(ROS_T.SHORT_IMP, 0) ROS_SHORT_IMP,
		ISNULL(PP_T.IMP_INC,0)PP_IMP_INC, ISNULL(TOS_T.IMP_INC,0)TOS_IMP_INC, ISNULL(ROS_T.IMP_INC,0)ROS_IMP_INC
	FROM MAIN_TAB MT LEFT JOIN (
		SELECT * FROM JOINED_TAB1 WHERE PLACEMENT='DETAIL PAGE ON-AMAZON') PP_T ON MT.CAMPAIGNID=PP_T.CAMPAIGN_ID
		LEFT JOIN (SELECT * FROM JOINED_TAB1 WHERE PLACEMENT='OTHER ON-AMAZON')ROS_T ON MT.CAMPAIGNID=ROS_T.CAMPAIGN_ID
		LEFT JOIN (SELECT * FROM JOINED_TAB1 WHERE PLACEMENT='TOP OF SEARCH ON-AMAZON')TOS_T ON MT.CAMPAIGNID=TOS_T.CAMPAIGN_ID
),
MERGED_WITH_CONV AS (
	SELECT MT.*, ISNULL(CVT.UNITS,0) AS UNITS, ISNULL(CST.COST,0) SPENTBUDGET
	FROM MERGED_TAB MT LEFT JOIN COST_TAB CST ON MT.CAMPAIGNID = CST.CAMPAIGN_ID
	LEFT JOIN CONV_TAB CVT ON 1=1
)
SELECT * 
FROM MERGED_WITH_CONV
WHERE @MONITOR = 0 OR (TOS_DELTA_CTR <= 0 OR ROS_DELTA_CTR <= 0 OR PP_DELTA_CTR <= 0)
        """
    )
    if not sql_data:
        with open("sp_deepq.txt", "a") as f:
            f.write(pst_time_str() + "\n" + "No Data!" + "\n\n")
        return None, [[0, 0, 0] for _ in range(100)]

    bufferFlag = sql_data[0]["BUFFERFLAG"]
    dynamic_explore = sql_data[0]["DYNAMICEXPLORE"]
    learning_rate = sql_data[0]["LR"]
    gamma = sql_data[0]["GAMMA"]
    epsilon = (
        sql_data[0]["EPSTART"]
        if not dynamic_explore
        else round(
            (min(0.99, sql_data[0]["EPSTART"]) ** (minute_from_last_reset / 60)), 2
        )
    )
    boltz_switch = sql_data[0]["BOLTZFLAG"]
    t_decay = sql_data[0]["TDECAY"]

    campaign_detail_dict = get_sp_spt_campaign_detail(
        [r["CAMPAIGNID"] for r in sql_data]
    )
    bid_dict = get_products_bids([r["CAMPAIGNID"] for r in sql_data])

    total_imp = 0
    tos_data, pp_data, ros_data = [], [], []

    for r in sql_data:
        all_plc_imp = r["TOS_IMP"] + r["PP_IMP"] + r["ROS_IMP"]
        total_imp += all_plc_imp
        if r["DEEPTOSSWITCH"]:
            tos_data.append(
                {
                    "CAMPAIGNID": r["CAMPAIGNID"],
                    "IMP": r["TOS_IMP"],
                    "CURR_CTR": r["TOS_CURR_CTR"],
                    "DELTA_CTR": r["TOS_DELTA_CTR"],
                    "CPC": r["TOS_CPC"],
                    "BIDLOWER": r["TOSBIDLOWER"],
                    "BIDUPPER": r["TOSBIDUPPER"],
                    "CVR": r["TOS_CVR"],
                    "CLICK_STEP": r["TOSCLICKSTEP"],
                    "CVRFLAG": r["CVRFLAG"],
                    "UNITS": r["UNITS"],
                    "COMBINEDDEEPQSWITCH": r["COMBINEDDEEPQSWITCH"],
                    "WEIGHTFLAG": r["WEIGHTFLAG"],
                    "CTRWEIGHT": r["CTRWEIGHT"],
                    "CONVWEIGHT": r["CONVWEIGHT"],
                    "CVRWEIGHT": r["CVRWEIGHT"],
                    "CPCWEIGHT": r["CPCWEIGHT"],
                    "BUDGETWEIGHT": r["BUDGETWEIGHT"],
                    "CURR_CPC": r["TOS_CURR_CPC"],
                    "PREV_CPC": r["TOS_PREV_CPC"],
                    "BASEBID": bid_dict.get(r["CAMPAIGNID"], 0),
                    "BOOST": campaign_detail_dict.get(r["CAMPAIGNID"], {}).get(
                        "TOS", 0
                    ),
                    "TOTALBID": get_totalbid_from_boost(
                        bid_dict.get(r["CAMPAIGNID"], 0),
                        campaign_detail_dict.get(r["CAMPAIGNID"], {}).get("TOS", 0),
                    ),
                    "STRATEGY": campaign_detail_dict.get(r["CAMPAIGNID"], {}).get(
                        "strategy", "LEGACY_FOR_SALES"
                    ),
                    "BUDGET": campaign_detail_dict.get(r["CAMPAIGNID"], {}).get(
                        "budget", 1
                    ),
                    "CLICK": r["TOS_CLK"],
                    "COST": r["TOS_COST"],
                    "PREV_CLICK": r["TOS_PREV_CLK"],
                    "PREV_CONV": r["TOS_PREV_CONV"],
                    "CONV": r["TOS_CONV"],
                    "SPENTBUDGET": r["SPENTBUDGET"],
                    "ONLYDROP": r["ONLYDROP"],
                    "DROPIMPROVED": r["DROPIMPROVED"],
                    "ALL_PLC_IMP": all_plc_imp,
                    "CTR_SHR": r["TOS_CTR_SHR"],
                    "LONG_IMP": r["TOS_LONG_IMP"],
                    "SHORT_IMP": r["TOS_SHORT_IMP"],
                    "IMP_INC": r["TOS_IMP_INC"],
                }
            )
        if r["DEEPPPSWITCH"]:
            pp_data.append(
                {
                    "CAMPAIGNID": r["CAMPAIGNID"],
                    "IMP": r["PP_IMP"],
                    "CURR_CTR": r["PP_CURR_CTR"],
                    "DELTA_CTR": r["PP_DELTA_CTR"],
                    "CPC": r["PP_CPC"],
                    "BIDLOWER": r["PPBIDLOWER"],
                    "BIDUPPER": r["PPBIDUPPER"],
                    "CVR": r["PP_CVR"],
                    "CLICK_STEP": r["PPCLICKSTEP"],
                    "CVRFLAG": r["CVRFLAG"],
                    "UNITS": r["UNITS"],
                    "COMBINEDDEEPQSWITCH": r["COMBINEDDEEPQSWITCH"],
                    "WEIGHTFLAG": r["WEIGHTFLAG"],
                    "CTRWEIGHT": r["CTRWEIGHT"],
                    "CONVWEIGHT": r["CONVWEIGHT"],
                    "CVRWEIGHT": r["CVRWEIGHT"],
                    "CPCWEIGHT": r["CPCWEIGHT"],
                    "BUDGETWEIGHT": r["BUDGETWEIGHT"],
                    "CURR_CPC": r["PP_CURR_CPC"],
                    "PREV_CPC": r["PP_PREV_CPC"],
                    "BASEBID": bid_dict.get(r["CAMPAIGNID"], 0),
                    "BOOST": campaign_detail_dict.get(r["CAMPAIGNID"], {}).get("PP", 0),
                    "TOTALBID": get_totalbid_from_boost(
                        bid_dict.get(r["CAMPAIGNID"], 0),
                        campaign_detail_dict.get(r["CAMPAIGNID"], {}).get("PP", 0),
                    ),
                    "STRATEGY": campaign_detail_dict.get(r["CAMPAIGNID"], {}).get(
                        "strategy", "LEGACY_FOR_SALES"
                    ),
                    "BUDGET": campaign_detail_dict.get(r["CAMPAIGNID"], {}).get(
                        "budget", 1
                    ),
                    "CLICK": r["PP_CLK"],
                    "COST": r["PP_COST"],
                    "PREV_CLICK": r["PP_PREV_CLK"],
                    "PREV_CONV": r["PP_PREV_CONV"],
                    "CONV": r["PP_CONV"],
                    "SPENTBUDGET": r["SPENTBUDGET"],
                    "ONLYDROP": r["ONLYDROP"],
                    "DROPIMPROVED": r["DROPIMPROVED"],
                    "ALL_PLC_IMP": all_plc_imp,
                    "CTR_SHR": r["PP_CTR_SHR"],
                    "LONG_IMP": r["PP_LONG_IMP"],
                    "SHORT_IMP": r["PP_SHORT_IMP"],
                    "IMP_INC": r["PP_IMP_INC"],
                }
            )
        if r["DEEPROSSWITCH"]:
            ros_data.append(
                {
                    "CAMPAIGNID": r["CAMPAIGNID"],
                    "IMP": r["ROS_IMP"],
                    "CURR_CTR": r["ROS_CURR_CTR"],
                    "DELTA_CTR": r["ROS_DELTA_CTR"],
                    "CPC": r["ROS_CPC"],
                    "BIDLOWER": r["ROSBIDLOWER"],
                    "BIDUPPER": r["ROSBIDUPPER"],
                    "CVR": r["ROS_CVR"],
                    "UNITS": r["UNITS"],
                    "COMBINEDDEEPQSWITCH": r["COMBINEDDEEPQSWITCH"],
                    "WEIGHTFLAG": r["WEIGHTFLAG"],
                    "CTRWEIGHT": r["CTRWEIGHT"],
                    "CONVWEIGHT": r["CONVWEIGHT"],
                    "CVRWEIGHT": r["CVRWEIGHT"],
                    "CPCWEIGHT": r["CPCWEIGHT"],
                    "BUDGETWEIGHT": r["BUDGETWEIGHT"],
                    "CURR_CPC": r["ROS_CURR_CPC"],
                    "PREV_CPC": r["ROS_PREV_CPC"],
                    "CLICK_STEP": r["ROSCLICKSTEP"],
                    "CVRFLAG": r["CVRFLAG"],
                    "BASEBID": bid_dict.get(r["CAMPAIGNID"], 0),
                    "BOOST": campaign_detail_dict.get(r["CAMPAIGNID"], {}).get(
                        "ROS", 0
                    ),
                    "TOTALBID": get_totalbid_from_boost(
                        bid_dict.get(r["CAMPAIGNID"], 0),
                        campaign_detail_dict.get(r["CAMPAIGNID"], {}).get("ROS", 0),
                    ),
                    "STRATEGY": campaign_detail_dict.get(r["CAMPAIGNID"], {}).get(
                        "strategy", "LEGACY_FOR_SALES"
                    ),
                    "BUDGET": campaign_detail_dict.get(r["CAMPAIGNID"], {}).get(
                        "budget", 1
                    ),
                    "CLICK": r["ROS_CLK"],
                    "COST": r["ROS_COST"],
                    "PREV_CLICK": r["ROS_PREV_CLK"],
                    "PREV_CONV": r["ROS_PREV_CONV"],
                    "CONV": r["ROS_CONV"],
                    "SPENTBUDGET": r["SPENTBUDGET"],
                    "ONLYDROP": r["ONLYDROP"],
                    "DROPIMPROVED": r["DROPIMPROVED"],
                    "ALL_PLC_IMP": all_plc_imp,
                    "CTR_SHR": r["ROS_CTR_SHR"],
                    "LONG_IMP": r["ROS_LONG_IMP"],
                    "SHORT_IMP": r["ROS_SHORT_IMP"],
                    "IMP_INC": r["ROS_IMP_INC"],
                }
            )

    tos_update_dict, tos_log, tos_reward, tos_reward_dist, tos_loss_dict = (
        run_deep_q_for_sp_spt(
            data=tos_data,
            campaignType="SP",
            placement="TOS",
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            boltz_switch=boltz_switch,
            t_decay=t_decay,
            total_imp=None if bufferFlag else total_imp,
        )
    )
    pp_update_dict, pp_log, pp_reward, pp_reward_dist, pp_loss_dict = (
        run_deep_q_for_sp_spt(
            data=pp_data,
            campaignType="SP",
            placement="PP",
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            boltz_switch=boltz_switch,
            t_decay=t_decay,
            total_imp=None if bufferFlag else total_imp,
        )
    )
    ros_update_dict, ros_log, ros_reward, ros_reward_dist, ros_loss_dict = (
        run_deep_q_for_sp_spt(
            data=ros_data,
            campaignType="SP",
            placement="ROS",
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            boltz_switch=boltz_switch,
            t_decay=t_decay,
            total_imp=None if bufferFlag else total_imp,
        )
    )

    sp_reward = None
    sp_reward_list = []
    if tos_reward is not None:
        sp_reward_list.append(tos_reward)
    if pp_reward is not None:
        sp_reward_list.append(pp_reward)
    if ros_reward is not None:
        sp_reward_list.append(ros_reward)

    if sp_reward_list:
        sp_reward = sum(sp_reward_list) / len(sp_reward_list)

    result_dict = defaultdict(dict)
    for d in (tos_update_dict, pp_update_dict, ros_update_dict):
        for outer_key, inner_dict in d.items():
            result_dict[outer_key].update(inner_dict)

    final_boost_update_dict = dict(result_dict)
    boost_update_list = create_json_for_boost_update(final_boost_update_dict)

    if boost_update_list:
        sp.UPDATE_CAMPAIGNS_SP(boost_update_list)

    with open("sp_deepq.txt", "a") as f:
        f.write(
            pst_time_str()
            + "\n"
            + str(boost_update_list)
            + "\n"
            + f"TOS: {tos_log}"
            + "\n"
            + f"PP: {pp_log}"
            + "\n"
            + f"ROS: {ros_log}"
            + "\n"
            + f"TOS_loss: {tos_loss_dict}"
            + "\n"
            + f"PP_loss: {pp_loss_dict}"
            + "\n"
            + f"ROS_loss: {ros_loss_dict}"
            + "\n\n"
        )

    return sp_reward, add_distributions(
        tos_reward_dist, pp_reward_dist, ros_reward_dist
    )


def deep_q_sp_bid(interval: int, monitor: bool):

    # Fetching data from SQL
    sql_data = qcm.fetch(
        f"""
--FOR SP BID
DECLARE @NOW DATETIME = '{pst_time_str()}',
	@INTERVAL INT = {interval},
	@MONITOR BIT = {int(monitor)};
DECLARE @STARTOFHOUR DATETIME2 = CAST(FORMAT(@NOW, 'yyyy-MM-dd HH:00') AS DATETIME2);
DECLARE @CVRDAYCOUNT INT = (SELECT MAX(CVRDAYCOUNT) FROM MODELING_DB.DBO.Q_MASTER
    WHERE DEEPQSWITCH = 1 AND CAMPAIGNTYPE IN ('SP'));

WITH MAIN_TAB AS (
    SELECT * FROM MODELING_DB.DBO.Q_MASTER
    WHERE DEEPQSWITCH = 1 AND CAMPAIGNTYPE IN ('SP')),
KEY_TAB AS (
        SELECT * FROM MODELING_DB.DBO.KEYWORDS_FOR_AGENT
        WHERE CAMPAIGNID IN (SELECT CAMPAIGNID FROM MAIN_TAB)),
SQS_TAB AS(
	SELECT CAMPAIGN_ID, KEYWORD_ID, PLACEMENT, CLICKS, IMPRESSIONS, COST, [TIMESTAMP]
	FROM [AMAZON_MARKETING].[DBO].AS_SP_TRAFFIC
	WHERE CAST(TIME_WINDOW_START AS DATE) = CAST(@NOW AS DATE)
		 AND CAMPAIGN_ID IN (SELECT CAMPAIGNID FROM MAIN_TAB)),
CURR_TAB AS (
    SELECT KEYWORD_ID, SUM(CLICKS) CLK0, SUM(IMPRESSIONS) IMP0, SUM(COST) COST0,
		CASE WHEN SUM(CLICKS) = 0 THEN 0 ELSE CAST(SUM(COST) AS FLOAT)/SUM(CLICKS) END AS CPC,
        CASE WHEN SUM(IMPRESSIONS) = 0 THEN 0 WHEN SUM(IMPRESSIONS) > 0 AND SUM(CLICKS) <= 0 THEN (1.0/SUM(IMPRESSIONS) - 1)
        ELSE CAST(SUM(CLICKS) AS FLOAT)/SUM(IMPRESSIONS) END AS CTR0
    FROM SQS_TAB GROUP BY KEYWORD_ID),
PREV_TAB AS (
    SELECT KEYWORD_ID, SUM(CLICKS) CLK1, SUM(IMPRESSIONS) IMP1,
		CASE WHEN SUM(CLICKS) = 0 THEN 0 ELSE CAST(SUM(COST) AS FLOAT)/SUM(CLICKS) END AS CPC1,
        CASE WHEN SUM(IMPRESSIONS) = 0 THEN 0 WHEN SUM(IMPRESSIONS) > 0 AND SUM(CLICKS) <= 0 THEN (1.0/SUM(IMPRESSIONS) - 1)
		ELSE CAST(SUM(CLICKS) AS FLOAT)/SUM(IMPRESSIONS) END AS CTR1
    FROM SQS_TAB WHERE [TIMESTAMP] <= DATEADD(MINUTE, -@INTERVAL, @NOW)
    GROUP BY KEYWORD_ID),
TOTAL_CLICK_TAB AS (
	SELECT KEYWORD_ID, SUM(CLICKS) AS CLICKS FROM [AMAZON_MARKETING].[DBO].AS_SP_TRAFFIC
	WHERE CAST(TIME_WINDOW_START AS DATETIME2) >= DATEADD(HOUR, -@CVRDAYCOUNT, @NOW) AND CAST(TIMESTAMP AS DATE) <= @NOW
	GROUP BY KEYWORD_ID),
TOTAL_CONV_TAB AS (
	SELECT KEYWORD_ID, SUM(CONVERSIONS_14D) AS CONVERSION FROM [AMAZON_MARKETING].[DBO].AS_SP_CONVERSION
	WHERE CAST(TIME_WINDOW_START AS DATETIME2) >= DATEADD(HOUR, -@CVRDAYCOUNT, @NOW) AND CAST(TIMESTAMP AS DATE) <= @NOW
	GROUP BY KEYWORD_ID),
CONV_TAB AS (
		SELECT SUM(CONVERSION) AS UNITS
		FROM (
			SELECT ISNULL(ORDERCOUNT - LAG(ORDERCOUNT) OVER (PARTITION BY CAST(TIMESTAMP AS DATE) ORDER BY TIMESTAMP), 0) AS CONVERSION
			FROM [AMAZON_MARKETING].[DBO].[AMZN_SALES_DATA]
			WHERE TIMESTAMP >= DATEADD(HOUR, -1, @STARTOFHOUR)
			  AND TIMESTAMP <= @NOW
		) AS TAB),
LATEST_CPC_TAB AS (
	SELECT KEYWORD_ID, MAX([TIMESTAMP]) LATEST
	FROM SQS_TAB WHERE COST > 0 GROUP BY KEYWORD_ID),
CPC_CURR_TAB AS(
	SELECT ST.KEYWORD_ID, SUM(CAST(ST.COST AS FLOAT)) / SUM(ST.CLICKS) AS CPC FROM
	SQS_TAB ST JOIN LATEST_CPC_TAB LCT ON ST.KEYWORD_ID = LCT.KEYWORD_ID AND ST.[TIMESTAMP] = LCT.LATEST AND ST.CLICKS > 0
	GROUP BY ST.KEYWORD_ID),
CPC_PREV_TAB AS(
	SELECT ST.KEYWORD_ID, SUM(CAST(ST.COST AS FLOAT)) / SUM(ST.CLICKS) AS CPC FROM
	SQS_TAB ST JOIN(
		SELECT ST.KEYWORD_ID, MAX(ST.[TIMESTAMP]) PREV
		FROM SQS_TAB ST JOIN LATEST_CPC_TAB LCT
			ON ST.KEYWORD_ID=LCT.KEYWORD_ID AND ST.[TIMESTAMP] < LCT.LATEST AND ST.CLICKS > 0
		GROUP BY ST.KEYWORD_ID
	) PCT ON ST.KEYWORD_ID = PCT.KEYWORD_ID AND ST.[TIMESTAMP] = PCT.PREV AND ST.CLICKS > 0
	GROUP BY ST.KEYWORD_ID),
COST_TAB AS (
	SELECT CAMPAIGN_ID, SUM(CAST(COST AS FLOAT)) COST FROM SQS_TAB GROUP BY CAMPAIGN_ID),
KEY_CONV_TAB AS (
	SELECT KEYWORD_ID, SUM(CONVERSIONS_14D) AS CONV FROM [AMAZON_MARKETING].[DBO].AS_SP_CONVERSION
	WHERE CAST(TIME_WINDOW_START AS DATE) = CAST(@NOW AS DATE) AND [TIMESTAMP] <= @NOW
		AND CAMPAIGN_ID IN (SELECT CAMPAIGNID FROM MAIN_TAB)
	GROUP BY KEYWORD_ID),
PREV_KEY_CONV_TAB AS (
	SELECT KEYWORD_ID, SUM(CONVERSIONS_14D) AS PREV_CONV FROM [AMAZON_MARKETING].[DBO].AS_SP_CONVERSION
	WHERE CAST(TIME_WINDOW_START AS DATE) = CAST(@NOW AS DATE) AND [TIMESTAMP] <= DATEADD(MINUTE, -@INTERVAL, @NOW)
		AND CAMPAIGN_ID IN (SELECT CAMPAIGNID FROM MAIN_TAB)
	GROUP BY KEYWORD_ID),
FINAL_TAB AS (
	SELECT KT.KEYWORDID, KT.CAMPAIGNID, KT.ADGROUPID, MT.GAMMA, MT.LR, MT.EPSTART, MT.BASEBIDLOWER, MT.BASEBIDUPPER, MT.COMBINEDDEEPQSWITCH,
		MT.CVRFLAG, MT.CLICKSTEP, CASE WHEN TCV.CONVERSION>0 THEN CAST (TCLK.CLICKS AS FLOAT) / TCV.CONVERSION ELSE 0 END CVR,
        MT.CTRWEIGHT, MT.CONVWEIGHT, MT.CVRWEIGHT, MT.WEIGHTFLAG, MT.CPCWEIGHT, MT.BUDGETWEIGHT, MT.DYNAMICEXPLORE,
        MT.BOLTZFLAG, MT.TDECAY,
		ISNULL(ROUND(CT.CTR0,4),0) AS CURR_CTR, ROUND(ISNULL(CT.CTR0,0) - ISNULL(PT.CTR1,0),4) AS DELTA_CTR,
		ISNULL(ROUND(CT.CPC,2),0) AS CPC, ISNULL(ROUND(PT.CPC1,2),0) AS CPC1, ISNULL(CVT.UNITS,0) UNITS,
        ISNULL(CCT.CPC,0) CURR_CPC, ISNULL(CPT.CPC,0) PREV_CPC, ISNULL(CST.COST,0) SPENTBUDGET, ISNULL(KCT.CONV,0) CONV,
        ISNULL(PKCT.PREV_CONV,0) PREV_CONV, ISNULL(PT.CLK1,0) PREV_CLICK, ISNULL(CT.CLK0,0) CLICK, ISNULL(CT.COST0,0) COST
	FROM KEY_TAB KT LEFT JOIN CURR_TAB CT ON KT.KEYWORDID = CT.KEYWORD_ID
		LEFT JOIN MAIN_TAB MT ON KT.CAMPAIGNID = MT.CAMPAIGNID
		LEFT JOIN PREV_TAB PT ON KT.KEYWORDID = PT.KEYWORD_ID
		LEFT JOIN TOTAL_CLICK_TAB TCLK ON KT.KEYWORDID = TCLK.KEYWORD_ID
		LEFT JOIN TOTAL_CONV_TAB TCV ON KT.KEYWORDID = TCV.KEYWORD_ID
        LEFT JOIN CPC_CURR_TAB CCT ON KT.KEYWORDID = CCT.KEYWORD_ID
		LEFT JOIN CPC_PREV_TAB CPT ON KT.KEYWORDID = CPT.KEYWORD_ID
		LEFT JOIN COST_TAB CST ON KT.CAMPAIGNID = CST.CAMPAIGN_ID
		LEFT JOIN KEY_CONV_TAB KCT ON KT.KEYWORDID = KCT.KEYWORD_ID
		LEFT JOIN PREV_KEY_CONV_TAB PKCT ON KT.KEYWORDID = PKCT.KEYWORD_ID
        LEFT JOIN CONV_TAB CVT ON 1=1)

SELECT * FROM FINAL_TAB
WHERE @MONITOR = 0 OR DELTA_CTR <= 0 --OR CPC <> CPC1
"""
    )
    if sql_data == []:
        with open("sp_bid_deepq.txt", "a") as f:
            f.write(pst_time_str() + "\n" + "No Data!" + "\n\n")
        return

    dynamic_explore = sql_data[0]["DYNAMICEXPLORE"]
    learning_rate = sql_data[0]["LR"]
    gamma = sql_data[0]["GAMMA"]
    epsilon = (
        sql_data[0]["EPSTART"]
        if not dynamic_explore
        else round(
            (min(0.99, sql_data[0]["EPSTART"]) ** (minute_from_last_reset / 60)), 2
        )
    )
    units = sql_data[0]["UNITS"]
    boltz_switch = sql_data[0]["BOLTZFLAG"]
    t_decay = sql_data[0]["TDECAY"]

    # Configuring the paths
    replay_buffer_path = path_dict["SP_BID"]["replay_buffer_path"]
    temp_buffer_path = path_dict["SP_BID"]["temp_buffer_path"]
    policy_model_path = path_dict["SP_BID"]["policy_model_path"]
    target_model_path = path_dict["SP_BID"]["target_model_path"]

    global_replay_buffer_path = path_dict["global"]["replay_buffer_path"]
    global_temp_buffer_path = path_dict["global"]["temp_buffer_path"]
    global_policy_model_path = path_dict["global"]["policy_model_path"]
    global_target_model_path = path_dict["global"]["target_model_path"]

    # Getting current bids from Amazon
    bid_dict = get_products_bids_byKey([k["KEYWORDID"] for k in sql_data])
    # Getting currently enabled campaignIds
    enabled_campaignids = get_sp_spt_campaign_detail(
        list({c["CAMPAIGNID"] for c in sql_data})
    )

    # ===>> Readying the data for updating pending buffers
    pending_buffer_update_data = {
        r["KEYWORDID"]: {
            "next_state": torch.tensor(
                [
                    r["CURR_CTR"],
                    r["DELTA_CTR"],
                    r["CPC"],
                    bid_dict.get(r["KEYWORDID"], 0),
                    r["PREV_CPC"] - r["CURR_CPC"],
                    enabled_campaignids.get(r["CAMPAIGNID"], {}).get("budget", 1)
                    - r["SPENTBUDGET"],
                ],
                dtype=torch.float32,
            ),
            "reward": r["CTRWEIGHT"] * r["DELTA_CTR"]
            + (
                r["BUDGETWEIGHT"]
                * (
                    negative_mapping(r["CONV"], r["CLICK"])
                    - negative_mapping(r["PREV_CONV"], r["PREV_CLICK"])
                )
            ),
        }
        for r in sql_data
        if bid_dict.get(r["KEYWORDID"], 0) >= 0.3
        and r["CAMPAIGNID"] in enabled_campaignids
    }
    global_pending_buffer_update_data = {}

    # Creating temporary buffer instant & updating the pending values
    temp_buffer = PersistentReplayBuffer(temp_buffer_path)
    temp_buffer.update(replay_buffer_path, pending_buffer_update_data)

    if policy_model_path == "./":
        dimensions = [6, 64, 64, 64, 64, 64, 64, 79]
        policy_model_path += (
            "starting_model_sp_bid_" + config_time_for_file_name() + ".pth"
        )
    else:
        dimensions = get_dimensions_from_file("dqn_models", policy_model_path)

    if global_policy_model_path == "./":
        global_dimensions = [6, 64, 64, 64, 64, 64, 64, 79]
        global_policy_model_path += (
            "starting_model_global_" + config_time_for_file_name() + ".pth"
        )
    else:
        global_dimensions = get_dimensions_from_file(
            "dqn_models", global_policy_model_path
        )

    # ===>> Creating DQN agent
    dqn_agent = DQNAgent(
        dimensions=dimensions,
        replay_path=replay_buffer_path,
        temp_replay_path=temp_buffer_path,
        policy_net_path=policy_model_path,
        target_net_path=target_model_path,
        epsilon_start=epsilon,
        gamma=gamma,
        lr=learning_rate,
        campaignType="SP_BID",
        boltz_switch=boltz_switch,
        t_decay=t_decay,
        t_start=1,
        t_end=0.1,
    )

    global_dqn_agent = DQNAgent(
        dimensions=global_dimensions,
        replay_path=global_replay_buffer_path,
        temp_replay_path=global_temp_buffer_path,
        policy_net_path=global_policy_model_path,
        target_net_path=global_target_model_path,
        lr=learning_rate,
        epsilon_start=epsilon,
        gamma=gamma,
        boltz_switch=boltz_switch,
        t_decay=t_decay,
        t_start=1,
        t_end=0.1,
    )
    loss_log = []

    # Training the network for 5 iterations
    for _ in range(5):
        loss_1 = dqn_agent.train_step()
        loss_2 = global_dqn_agent.train_step()
        if loss_1 is not None:
            loss_log.append({"agent": "SP_bid", "loss": loss_1})
        if loss_2 is not None:
            loss_log.append({"agent": "GLOBAL", "loss": loss_2})
    # Saving models
    dqn_agent.save_models()
    global_dqn_agent.save_models()

    # ===>> Taking actions
    bid_update_list = []
    temp_buffer_list = []
    log = []

    for row in sql_data:
        keywordId = row["KEYWORDID"]
        adGroupId = row["ADGROUPID"]
        campaignId = row["CAMPAIGNID"]
        bid = bid_dict.get(keywordId, 0)
        cvr_condition = row["CVRFLAG"]
        global_deep_q = row["COMBINEDDEEPQSWITCH"]
        weight_flag = row["WEIGHTFLAG"]
        del_cpc = row["PREV_CPC"] - row["CURR_CPC"]
        budget = enabled_campaignids.get(campaignId, {}).get("budget", 1)

        max_cvr = 100
        if row["CVR"] <= row["CLICKSTEP"]:
            normalized_cvr = 0
        elif row["CVR"] >= 100:
            normalized_cvr = 1
        else:
            normalized_cvr = (row["CVR"] - row["CLICKSTEP"]) / (
                max_cvr - row["CLICKSTEP"]
            )

        if bid_dict.get(keywordId, 0) >= 0.3 and campaignId in enabled_campaignids:
            global_pending_buffer_update_data.update(
                {
                    keywordId: {
                        "next_state": torch.tensor(
                            [
                                row["CURR_CTR"],
                                row["DELTA_CTR"],
                                row["CPC"],
                                bid_dict.get(keywordId, 0),
                                del_cpc,
                                budget - row["SPENTBUDGET"],
                            ],
                            dtype=torch.float32,
                        ),
                        "reward": (
                            (row["CTRWEIGHT"] * row["DELTA_CTR"])
                            + (row["CONVWEIGHT"] * units / 10)
                            + (
                                row["CPCWEIGHT"]
                                * (
                                    del_cpc / row["PREV_CPC"]
                                    if row["PREV_CPC"] > 0
                                    else 0
                                )
                            )
                            + (
                                row["CVRWEIGHT"]
                                * negative_mapping(row["CONV"], row["CLICK"])
                            )
                            # + (row["BUDGETWEIGHT"] * (budget - row["COST"]) / budget)
                            + (
                                row["BUDGETWEIGHT"]
                                * (
                                    negative_mapping(row["CONV"], row["CLICK"])
                                    - negative_mapping(
                                        row["PREV_CONV"], row["PREV_CLICK"]
                                    )
                                )
                            )
                            if weight_flag
                            else (row["DELTA_CTR"] + units / 10)
                        ),
                    }
                }
            )

        # Only taking actions if bid >= 0.3
        if bid >= 0.3 and campaignId in enabled_campaignids:
            action = (
                dqn_agent.select_action(
                    [
                        row["CURR_CTR"],
                        row["DELTA_CTR"],
                        row["CPC"],
                        bid,
                        del_cpc,
                        budget - row["SPENTBUDGET"],
                    ]
                )
                if not global_deep_q
                else global_dqn_agent.select_action(
                    [
                        row["CURR_CTR"],
                        row["DELTA_CTR"],
                        row["CPC"],
                        bid,
                        del_cpc,
                        budget - row["SPENTBUDGET"],
                    ]
                )
            )
            calculated_bid = bid_from_action(0.3, 0.2, action)

            # Sick joke of bounding the bid
            capped_bid = round(
                min(max(calculated_bid, row["BASEBIDLOWER"]), row["BASEBIDUPPER"]), 2
            )
            capped_action = action_from_bid(0.3, 0.2, capped_bid)

            if cvr_condition:
                if random.uniform(0, 1) > normalized_cvr:
                    app_bid = capped_bid
                else:
                    app_bid = 0.3
                    capped_action = 0
                    log.append(
                        {
                            "keywordId": keywordId,
                            "cvrn": True,
                            "prevBid": bid,
                        }
                    )

            else:
                app_bid = capped_bid

            # For updating bid in Amazon
            bid_update_list.append(
                {
                    "campaignId": campaignId,
                    "adGroupId": adGroupId,
                    "keywordId": keywordId,
                    "bid": app_bid,
                }
            )
            # Creating buffer for new buffer acquisition in Temporary Buffer file
            temp_buffer_list.append(
                {
                    "unique_id": keywordId,
                    "state": torch.tensor(
                        [
                            row["CURR_CTR"],
                            row["DELTA_CTR"],
                            row["CPC"],
                            bid,
                            del_cpc,
                            budget - row["SPENTBUDGET"],
                        ],
                        dtype=torch.float32,
                    ),
                    "action": capped_action,
                    "next_state": None,
                    "reward": None,
                    "done": False,
                }
            )

    global_temp_buffer = PersistentReplayBuffer(global_temp_buffer_path)
    global_temp_buffer.update(
        global_replay_buffer_path, global_pending_buffer_update_data
    )

    if bid_update_list:
        sp.UPDATE_KEYWORDS_SP(bid_update_list)

    if temp_buffer_list:
        temp_buffer.store(temp_buffer_list)
        global_temp_buffer.store(temp_buffer_list)

    batch_rewards = [v["reward"] for v in global_pending_buffer_update_data.values()]
    avg_reward = sum(batch_rewards) / len(batch_rewards) if batch_rewards else None

    individual_loss = [item["loss"] for item in loss_log if item["agent"] == "SP_bid"]
    avg_sv_loss = sum(individual_loss) / len(individual_loss) if individual_loss else 0

    global_loss = [item["loss"] for item in loss_log if item["agent"] == "GLOBAL"]
    avg_global_loss = sum(global_loss) / len(global_loss) if global_loss else 0

    loss_dict = {"Individual": avg_sv_loss, "Global": avg_global_loss}

    with open("sp_bid_deepq.txt", "a") as f:
        f.write(
            pst_time_str()
            + "\n"
            + str(bid_update_list)
            + "\n"
            + str(log)
            + "\n"
            + str(loss_dict)
            + "\n\n"
        )

    return avg_reward


def deep_q_spt(interval: int, monitor: bool):
    sql_data = qcm.fetch(
        f"""
--FOR SPT
DECLARE @NOW DATETIME = '{pst_time_str()}',
	@INTERVAL INT = {interval},
	@MONITOR BIT = {int(monitor)};
DECLARE @STARTOFHOUR DATETIME2 = CAST(FORMAT(@NOW, 'yyyy-MM-dd HH:00') AS DATETIME2);
DECLARE @CVRDAYCOUNT INT = (SELECT MAX(CVRDAYCOUNT) FROM MODELING_DB.DBO.Q_MASTER
    WHERE (DEEPTOSSWITCH = 1 OR DEEPPPSWITCH = 1 OR DEEPROSSWITCH = 1) AND CAMPAIGNTYPE IN ('SPT'));
DECLARE @IMPVOLUMEHOUR INT = (SELECT MAX(IMPVOLUMEHOUR) FROM MODELING_DB.DBO.Q_MASTER
    WHERE (DEEPTOSSWITCH = 1 OR DEEPPPSWITCH = 1 OR DEEPROSSWITCH = 1) AND CAMPAIGNTYPE IN ('SPT'));
DECLARE @SHORTIMPHOUR INT = (SELECT MAX(SHORTIMPHOUR) FROM MODELING_DB.DBO.Q_MASTER
    WHERE (DEEPTOSSWITCH = 1 OR DEEPPPSWITCH = 1 OR DEEPROSSWITCH = 1) AND CAMPAIGNTYPE IN ('SPT'));
DECLARE @ITERATIONCOUNT INT = (SELECT MAX(ITERATIONCOUNT) FROM MODELING_DB.DBO.Q_MASTER
    WHERE (DEEPTOSSWITCH = 1 OR DEEPPPSWITCH = 1 OR DEEPROSSWITCH = 1) AND CAMPAIGNTYPE IN ('SPT'));
DECLARE @IMPRESSIONTHRESHOLD INT = (SELECT MAX(IMPRESSIONTHRESHOLD) FROM MODELING_DB.DBO.Q_MASTER
    WHERE (DEEPTOSSWITCH = 1 OR DEEPPPSWITCH = 1 OR DEEPROSSWITCH = 1) AND CAMPAIGNTYPE IN ('SPT'));

WITH MAIN_TAB AS (
    SELECT * FROM MODELING_DB.DBO.Q_MASTER
    WHERE (DEEPTOSSWITCH = 1 OR DEEPPPSWITCH = 1 OR DEEPROSSWITCH = 1) AND CAMPAIGNTYPE IN ('SPT')),
SQS_TAB AS(
	SELECT CAMPAIGN_ID, PLACEMENT, CLICKS, IMPRESSIONS, COST, [TIMESTAMP]
	FROM [AMAZON_MARKETING].[DBO].AS_SP_TRAFFIC
	WHERE CAST(TIME_WINDOW_START AS DATE) = CAST(@NOW AS DATE)
		 AND CAMPAIGN_ID IN (SELECT CAMPAIGNID FROM MAIN_TAB)),
CURR_TAB AS (
    SELECT CAMPAIGN_ID, PLACEMENT, SUM(CLICKS) CLK0, SUM(IMPRESSIONS) IMP0, SUM(COST) COST0,
		CASE WHEN SUM(CLICKS) = 0 THEN 0 ELSE CAST(SUM(COST) AS FLOAT)/SUM(CLICKS) END AS CPC,
        CASE WHEN SUM(IMPRESSIONS) = 0 THEN 0 WHEN SUM(IMPRESSIONS) > 0 AND SUM(CLICKS) <= 0 THEN (1.0/SUM(IMPRESSIONS) - 1)
        ELSE CAST(SUM(CLICKS) AS FLOAT)/SUM(IMPRESSIONS) END AS CTR0
    FROM SQS_TAB GROUP BY CAMPAIGN_ID, PLACEMENT),
PREV_TAB AS (
    SELECT CAMPAIGN_ID, PLACEMENT, SUM(CLICKS) CLK1, SUM(IMPRESSIONS) IMP1,
		CASE WHEN SUM(CLICKS) = 0 THEN 0 ELSE CAST(SUM(COST) AS FLOAT)/SUM(CLICKS) END AS CPC1,
        CASE WHEN SUM(IMPRESSIONS) = 0 THEN 0 WHEN SUM(IMPRESSIONS) > 0 AND SUM(CLICKS) <= 0 THEN (1.0/SUM(IMPRESSIONS) - 1)
		ELSE CAST(SUM(CLICKS) AS FLOAT)/SUM(IMPRESSIONS) END AS CTR1
    FROM SQS_TAB WHERE [TIMESTAMP] <= DATEADD(MINUTE, -@INTERVAL, @NOW)
    GROUP BY CAMPAIGN_ID, PLACEMENT),
BAL_IMP_TAB AS(
	SELECT CAMPAIGN_ID, PLACEMENT, 100*(MAX(NOW_IMP) - MAX(BAL_ITER_IMP))/(MAX(BAL_ITER_IMP)+1) IMP_INC FROM(
		SELECT CAMPAIGN_ID, PLACEMENT, CAST(SUM(IMPRESSIONS)AS FLOAT)/@ITERATIONCOUNT BAL_ITER_IMP, 0 NOW_IMP
		FROM SQS_TAB WHERE IMPRESSIONS >= 0 AND TIMESTAMP BETWEEN DATEADD(MINUTE, -60*@ITERATIONCOUNT, @NOW) AND DATEADD(MINUTE, -60, @NOW)
		GROUP BY CAMPAIGN_ID, PLACEMENT
		UNION
		SELECT CAMPAIGN_ID, PLACEMENT, 0 BAL_ITER_IMP, CAST(SUM(IMPRESSIONS)AS FLOAT) NOW_IMP
		FROM SQS_TAB WHERE IMPRESSIONS >= 0 AND TIMESTAMP BETWEEN DATEADD(MINUTE, -60, @NOW) AND @NOW
		GROUP BY CAMPAIGN_ID, PLACEMENT)T
    GROUP BY CAMPAIGN_ID, PLACEMENT),
TOTAL_CLICK_TAB AS (
	SELECT CAMPAIGN_ID, PLACEMENT, SUM(CLICKS) AS CLICKS FROM [AMAZON_MARKETING].[DBO].AS_SP_TRAFFIC
	WHERE CAST(TIME_WINDOW_START AS DATETIME2) >= DATEADD(HOUR, -@CVRDAYCOUNT, @NOW) AND CAST(TIMESTAMP AS DATE) <= @NOW
	GROUP BY CAMPAIGN_ID, PLACEMENT),
LONG_IMP_TAB AS (
	SELECT CAMPAIGN_ID, PLACEMENT, CAST(IMP AS FLOAT) LONG_IMP
	FROM (SELECT CAMPAIGN_ID, PLACEMENT, SUM(IMPRESSIONS) AS IMP FROM [AMAZON_MARKETING].[DBO].AS_SP_TRAFFIC
		WHERE TIMESTAMP BETWEEN DATEADD(HOUR, -@IMPVOLUMEHOUR, @NOW) AND @NOW
		GROUP BY CAMPAIGN_ID, PLACEMENT)T),
SHORT_IMP_TAB AS (
	SELECT CAMPAIGN_ID, PLACEMENT, CAST(IMP AS FLOAT) SHORT_IMP
	FROM (SELECT CAMPAIGN_ID, PLACEMENT, SUM(IMPRESSIONS) AS IMP FROM [AMAZON_MARKETING].[DBO].AS_SP_TRAFFIC
		WHERE TIMESTAMP BETWEEN DATEADD(HOUR, -@SHORTIMPHOUR, @NOW) AND @NOW
		GROUP BY CAMPAIGN_ID, PLACEMENT)T),
TOTAL_CONV_TAB AS (
	SELECT CAMPAIGN_ID, PLACEMENT, SUM(CONVERSIONS_14D) AS CONVERSION FROM [AMAZON_MARKETING].[DBO].AS_SP_CONVERSION
	WHERE CAST(TIME_WINDOW_START AS DATETIME2) >= DATEADD(HOUR, -@CVRDAYCOUNT, @NOW) AND CAST(TIMESTAMP AS DATE) <= @NOW
	GROUP BY CAMPAIGN_ID, PLACEMENT),
CVR_TAB AS (
	SELECT TCLK.CAMPAIGN_ID, TCLK.PLACEMENT, TCLK.CLICKS, TCV.CONVERSION,
		CASE WHEN TCV.CONVERSION>0 THEN CAST (TCLK.CLICKS AS FLOAT) / TCV.CONVERSION ELSE 0 END CVR
	FROM TOTAL_CLICK_TAB TCLK LEFT JOIN TOTAL_CONV_TAB TCV
		ON TCLK.CAMPAIGN_ID = TCV.CAMPAIGN_ID AND TCLK.PLACEMENT = TCV.PLACEMENT),
LATEST_CPC_TAB AS (
	SELECT CAMPAIGN_ID, PLACEMENT, MAX([TIMESTAMP]) LATEST
	FROM SQS_TAB WHERE COST > 0 GROUP BY CAMPAIGN_ID, PLACEMENT),
CPC_CURR_TAB AS(
	SELECT ST.CAMPAIGN_ID, ST.PLACEMENT, SUM(CAST(ST.COST AS FLOAT)) / SUM(ST.CLICKS) AS CPC FROM
	SQS_TAB ST JOIN LATEST_CPC_TAB LCT ON ST.CAMPAIGN_ID = LCT.CAMPAIGN_ID AND ST.PLACEMENT = LCT.PLACEMENT AND ST.[TIMESTAMP] = LCT.LATEST AND ST.CLICKS > 0
	GROUP BY ST.CAMPAIGN_ID, ST.PLACEMENT),
CPC_PREV_TAB AS(
	SELECT ST.CAMPAIGN_ID, ST.PLACEMENT, SUM(CAST(ST.COST AS FLOAT)) / SUM(ST.CLICKS) AS CPC FROM
	SQS_TAB ST JOIN(
		SELECT ST.CAMPAIGN_ID, ST.PLACEMENT, MAX(ST.[TIMESTAMP]) PREV
		FROM SQS_TAB ST JOIN LATEST_CPC_TAB LCT
			ON ST.CAMPAIGN_ID=LCT.CAMPAIGN_ID AND ST.PLACEMENT=LCT.PLACEMENT AND ST.[TIMESTAMP] < LCT.LATEST AND ST.CLICKS > 0
		GROUP BY ST.CAMPAIGN_ID, ST.PLACEMENT
	) PCT ON ST.CAMPAIGN_ID = PCT.CAMPAIGN_ID AND ST.PLACEMENT = PCT.PLACEMENT AND ST.[TIMESTAMP] = PCT.PREV AND ST.CLICKS > 0
	GROUP BY ST.CAMPAIGN_ID, ST.PLACEMENT),
PLCMNT_CONV_TAB AS (
	SELECT CAMPAIGN_ID, PLACEMENT, SUM(CONVERSIONS_14D) AS CONV FROM [AMAZON_MARKETING].[DBO].AS_SP_CONVERSION
	WHERE CAST(TIME_WINDOW_START AS DATE) = CAST(@NOW AS DATE) AND [TIMESTAMP] <= @NOW
		AND CAMPAIGN_ID IN (SELECT CAMPAIGNID FROM MAIN_TAB)
	GROUP BY CAMPAIGN_ID, PLACEMENT),
PREV_PLCMNT_CONV_TAB AS (
	SELECT CAMPAIGN_ID, PLACEMENT, SUM(CONVERSIONS_14D) AS PREV_CONV FROM [AMAZON_MARKETING].[DBO].AS_SP_CONVERSION
	WHERE CAST(TIME_WINDOW_START AS DATE) = CAST(@NOW AS DATE) AND [TIMESTAMP] <= DATEADD(MINUTE, -@INTERVAL, @NOW)
		AND CAMPAIGN_ID IN (SELECT CAMPAIGNID FROM MAIN_TAB)
	GROUP BY CAMPAIGN_ID, PLACEMENT),
JOINED_TAB AS (
	SELECT CT.CAMPAIGN_ID, CT.PLACEMENT, CT.CLK0, CT.IMP0, CT.CTR0, CT.CPC, CAST(CT.COST0 AS FLOAT) COST0, LIT.LONG_IMP, SIT.SHORT_IMP,
		PT.CLK1, PT.IMP1, PT.CTR1, PT.CPC1, CVT.CVR,
		CASE WHEN ISNULL([BIT].IMP_INC, 0) < @IMPRESSIONTHRESHOLD THEN 0 ELSE ISNULL([BIT].IMP_INC, 0) END IMP_INC,
        CPC_C.CPC AS CURR_CPC, CPC_P.CPC AS PREV_CPC, PCT.CONV, PVCT.PREV_CONV
	FROM CURR_TAB CT LEFT JOIN PREV_TAB PT ON CT.CAMPAIGN_ID=PT.CAMPAIGN_ID AND CT.PLACEMENT=PT.PLACEMENT
		LEFT JOIN CVR_TAB CVT ON CT.CAMPAIGN_ID=CVT.CAMPAIGN_ID AND CT.PLACEMENT=CVT.PLACEMENT
        LEFT JOIN CPC_CURR_TAB CPC_C ON CT.CAMPAIGN_ID=CPC_C.CAMPAIGN_ID AND CT.PLACEMENT=CPC_C.PLACEMENT
		LEFT JOIN CPC_PREV_TAB CPC_P ON CT.CAMPAIGN_ID=CPC_P.CAMPAIGN_ID AND CT.PLACEMENT=CPC_P.PLACEMENT
        LEFT JOIN PREV_PLCMNT_CONV_TAB PVCT ON CT.CAMPAIGN_ID=PVCT.CAMPAIGN_ID AND CT.PLACEMENT=PVCT.PLACEMENT
		LEFT JOIN LONG_IMP_TAB LIT ON CT.CAMPAIGN_ID=LIT.CAMPAIGN_ID AND CT.PLACEMENT=LIT.PLACEMENT
		LEFT JOIN SHORT_IMP_TAB SIT ON CT.CAMPAIGN_ID=SIT.CAMPAIGN_ID AND CT.PLACEMENT=SIT.PLACEMENT
		LEFT JOIN PLCMNT_CONV_TAB PCT ON CT.CAMPAIGN_ID=PCT.CAMPAIGN_ID AND CT.PLACEMENT=PCT.PLACEMENT
		LEFT JOIN BAL_IMP_TAB [BIT] ON CT.CAMPAIGN_ID=[BIT].CAMPAIGN_ID AND CT.PLACEMENT = [BIT].PLACEMENT),
CONV_TAB AS (
		SELECT SUM(CONVERSION) AS UNITS
		FROM (
			SELECT ISNULL(ORDERCOUNT - LAG(ORDERCOUNT) OVER (PARTITION BY CAST(TIMESTAMP AS DATE) ORDER BY TIMESTAMP), 0) AS CONVERSION
			FROM [AMAZON_MARKETING].[DBO].[AMZN_SALES_DATA]
			WHERE TIMESTAMP >= DATEADD(HOUR, -1, @STARTOFHOUR)
			  AND TIMESTAMP <= @NOW
		) AS TAB),
COST_TAB AS (
	SELECT CAMPAIGN_ID, SUM(CAST(COST AS FLOAT)) COST FROM SQS_TAB GROUP BY CAMPAIGN_ID),
MERGED_TAB AS(
	SELECT MT.CAMPAIGNID,MT.GAMMA, MT.LR, MT.EPSTART, MT.DEEPTOSSWITCH, MT.DEEPPPSWITCH, MT.DEEPROSSWITCH, MT.COMBINEDDEEPQSWITCH,
		MT.TOSBIDLOWER, MT.TOSBIDUPPER, MT.PPBIDLOWER, MT.PPBIDUPPER, MT.ROSBIDLOWER, MT.ROSBIDUPPER, MT.CVRFLAG, MT.DYNAMICEXPLORE,
		MT.PPCLICKSTEP, MT.ROSCLICKSTEP, MT.TOSCLICKSTEP, ISNULL(PP_T.CVR,0) PP_CVR, ISNULL(ROS_T.CVR,0) ROS_CVR, ISNULL(TOS_T.CVR,0) TOS_CVR,
        MT.WEIGHTFLAG, MT.CTRWEIGHT, MT.CONVWEIGHT, MT.CVRWEIGHT, MT.CPCWEIGHT, MT.BUDGETWEIGHT,
        MT.BOLTZFLAG, MT.TDECAY,
		ISNULL(PP_T.CTR0,0) PP_CURR_CTR, ISNULL(ROS_T.CTR0,0) ROS_CURR_CTR, ISNULL(TOS_T.CTR0,0) TOS_CURR_CTR,
		ISNULL(ROUND(TOS_T.CTR0 - ISNULL(TOS_T.CTR1,0),4),0) TOS_DELTA_CTR,
		ISNULL(ROUND(PP_T.CTR0 - ISNULL(PP_T.CTR1,0),4),0) PP_DELTA_CTR, ISNULL(ROUND(ROS_T.CTR0 - ISNULL(ROS_T.CTR1,0),4),0) ROS_DELTA_CTR,
		ISNULL(PP_T.CPC,0) PP_CPC, ISNULL(ROS_T.CPC,0) ROS_CPC, ISNULL(TOS_T.CPC,0) TOS_CPC,
		ISNULL(PP_T.CPC1,0) PP_CPC1, ISNULL(ROS_T.CPC1,0) ROS_CPC1, ISNULL(TOS_T.CPC1,0) TOS_CPC1,
        ISNULL(PP_T.CURR_CPC,0) PP_CURR_CPC, ISNULL(ROS_T.CURR_CPC,0) ROS_CURR_CPC, ISNULL(TOS_T.CURR_CPC,0) TOS_CURR_CPC,
		ISNULL(PP_T.PREV_CPC,0) PP_PREV_CPC, ISNULL(ROS_T.PREV_CPC,0) ROS_PREV_CPC, ISNULL(TOS_T.PREV_CPC,0) TOS_PREV_CPC,
		ISNULL(PP_T.CLK1,0) PP_PREV_CLK, ISNULL(TOS_T.CLK1,0) TOS_PREV_CLK, ISNULL(ROS_T.CLK1,0) ROS_PREV_CLK,
		ISNULL(PP_T.CLK0,0) PP_CLK, ISNULL(TOS_T.CLK0,0) TOS_CLK, ISNULL(ROS_T.CLK0,0) ROS_CLK,
		ISNULL(PP_T.COST0,0) PP_COST, ISNULL(TOS_T.COST0,0) TOS_COST, ISNULL(ROS_T.COST0,0) ROS_COST,
		ISNULL(PP_T.PREV_CONV,0) PP_PREV_CONV, ISNULL(TOS_T.PREV_CONV,0) TOS_PREV_CONV, ISNULL(ROS_T.PREV_CONV,0) ROS_PREV_CONV,
		ISNULL(PP_T.CONV,0) PP_CONV, ISNULL(TOS_T.CONV,0) TOS_CONV, ISNULL(ROS_T.CONV,0) ROS_CONV,
		ISNULL(PP_T.LONG_IMP, 0) PP_LONG_IMP,  ISNULL(TOS_T.LONG_IMP, 0) TOS_LONG_IMP, ISNULL(ROS_T.LONG_IMP, 0) ROS_LONG_IMP,
		ISNULL(PP_T.SHORT_IMP, 0) PP_SHORT_IMP,  ISNULL(TOS_T.SHORT_IMP, 0) TOS_SHORT_IMP, ISNULL(ROS_T.SHORT_IMP, 0) ROS_SHORT_IMP,
		ISNULL(PP_T.IMP_INC,0)PP_IMP_INC, ISNULL(TOS_T.IMP_INC,0)TOS_IMP_INC, ISNULL(ROS_T.IMP_INC,0)ROS_IMP_INC
	FROM MAIN_TAB MT LEFT JOIN (
		SELECT * FROM JOINED_TAB WHERE PLACEMENT='DETAIL PAGE ON-AMAZON') PP_T ON MT.CAMPAIGNID=PP_T.CAMPAIGN_ID
		LEFT JOIN (SELECT * FROM JOINED_TAB WHERE PLACEMENT='OTHER ON-AMAZON')ROS_T ON MT.CAMPAIGNID=ROS_T.CAMPAIGN_ID
		LEFT JOIN (SELECT * FROM JOINED_TAB WHERE PLACEMENT='TOP OF SEARCH ON-AMAZON')TOS_T ON MT.CAMPAIGNID=TOS_T.CAMPAIGN_ID
),
MERGED_WITH_CONV AS (
	SELECT MT.*, ISNULL(CVT.UNITS,0) AS UNITS, ISNULL(CST.COST,0) SPENTBUDGET
	FROM MERGED_TAB MT LEFT JOIN COST_TAB CST ON MT.CAMPAIGNID = CST.CAMPAIGN_ID
	LEFT JOIN CONV_TAB CVT ON 1=1
)
SELECT * 
FROM MERGED_WITH_CONV
WHERE (@MONITOR = 0) OR (TOS_DELTA_CTR <= 0 OR ROS_DELTA_CTR <= 0 OR PP_DELTA_CTR <= 0)
	---OR (TOS_CPC <> TOS_CPC1 OR PP_CPC <> PP_CPC1 OR ROS_CPC <> ROS_CPC1)
        """
    )
    if not sql_data:
        with open("spt_deepq.txt", "a") as f:
            f.write(pst_time_str() + "\n" + "No Data!" + "\n\n")
        return None, [[0, 0, 0] for _ in range(100)]

    dynamic_explore = sql_data[0]["DYNAMICEXPLORE"]
    learning_rate = sql_data[0]["LR"]
    gamma = sql_data[0]["GAMMA"]
    epsilon = (
        sql_data[0]["EPSTART"]
        if not dynamic_explore
        else round(
            (min(0.99, sql_data[0]["EPSTART"]) ** (minute_from_last_reset / 60)), 2
        )
    )
    boltz_switch = sql_data[0]["BOLTZFLAG"]
    t_decay = sql_data[0]["TDECAY"]

    campaign_detail_dict = get_sp_spt_campaign_detail(
        [r["CAMPAIGNID"] for r in sql_data]
    )
    bid_dict = get_pat_bids([r["CAMPAIGNID"] for r in sql_data])

    tos_data = [
        {
            "CAMPAIGNID": r["CAMPAIGNID"],
            "CURR_CTR": r["TOS_CURR_CTR"],
            "DELTA_CTR": r["TOS_DELTA_CTR"],
            "CPC": r["TOS_CPC"],
            "BIDLOWER": r["TOSBIDLOWER"],
            "BIDUPPER": r["TOSBIDUPPER"],
            "CVR": r["TOS_CVR"],
            "CLICK_STEP": r["TOSCLICKSTEP"],
            "CVRFLAG": r["CVRFLAG"],
            "UNITS": r["UNITS"],
            "COMBINEDDEEPQSWITCH": r["COMBINEDDEEPQSWITCH"],
            "WEIGHTFLAG": r["WEIGHTFLAG"],
            "CTRWEIGHT": r["CTRWEIGHT"],
            "CONVWEIGHT": r["CONVWEIGHT"],
            "CVRWEIGHT": r["CVRWEIGHT"],
            "CPCWEIGHT": r["CPCWEIGHT"],
            "BUDGETWEIGHT": r["BUDGETWEIGHT"],
            "CURR_CPC": r["TOS_CURR_CPC"],
            "PREV_CPC": r["TOS_PREV_CPC"],
            "BASEBID": bid_dict.get(r["CAMPAIGNID"], 0),
            "BOOST": campaign_detail_dict.get(r["CAMPAIGNID"], {}).get("TOS", 0),
            "TOTALBID": get_totalbid_from_boost(
                bid_dict.get(r["CAMPAIGNID"], 0),
                campaign_detail_dict.get(r["CAMPAIGNID"], {}).get("TOS", 0),
            ),
            "STRATEGY": campaign_detail_dict.get(r["CAMPAIGNID"], {}).get(
                "strategy", "LEGACY_FOR_SALES"
            ),
            "BUDGET": campaign_detail_dict.get(r["CAMPAIGNID"], {}).get("budget", 1),
            "CLICK": r["TOS_CLK"],
            "COST": r["TOS_COST"],
            "PREV_CLICK": r["TOS_PREV_CLK"],
            "PREV_CONV": r["TOS_PREV_CONV"],
            "CONV": r["TOS_CONV"],
            "SPENTBUDGET": r["SPENTBUDGET"],
            "LONG_IMP": r["TOS_LONG_IMP"],
            "SHORT_IMP": r["TOS_SHORT_IMP"],
            "IMP_INC": r["TOS_IMP_INC"],
        }
        for r in sql_data
        if r["DEEPTOSSWITCH"]
    ]
    pp_data = [
        {
            "CAMPAIGNID": r["CAMPAIGNID"],
            "CURR_CTR": r["PP_CURR_CTR"],
            "DELTA_CTR": r["PP_DELTA_CTR"],
            "CPC": r["PP_CPC"],
            "BIDLOWER": r["PPBIDLOWER"],
            "BIDUPPER": r["PPBIDUPPER"],
            "CVR": r["PP_CVR"],
            "CLICK_STEP": r["PPCLICKSTEP"],
            "CVRFLAG": r["CVRFLAG"],
            "UNITS": r["UNITS"],
            "COMBINEDDEEPQSWITCH": r["COMBINEDDEEPQSWITCH"],
            "WEIGHTFLAG": r["WEIGHTFLAG"],
            "CTRWEIGHT": r["CTRWEIGHT"],
            "CONVWEIGHT": r["CONVWEIGHT"],
            "CVRWEIGHT": r["CVRWEIGHT"],
            "CPCWEIGHT": r["CPCWEIGHT"],
            "BUDGETWEIGHT": r["BUDGETWEIGHT"],
            "CURR_CPC": r["PP_CURR_CPC"],
            "PREV_CPC": r["PP_PREV_CPC"],
            "BASEBID": bid_dict.get(r["CAMPAIGNID"], 0),
            "BOOST": campaign_detail_dict.get(r["CAMPAIGNID"], {}).get("PP", 0),
            "TOTALBID": get_totalbid_from_boost(
                bid_dict.get(r["CAMPAIGNID"], 0),
                campaign_detail_dict.get(r["CAMPAIGNID"], {}).get("PP", 0),
            ),
            "STRATEGY": campaign_detail_dict.get(r["CAMPAIGNID"], {}).get(
                "strategy", "LEGACY_FOR_SALES"
            ),
            "BUDGET": campaign_detail_dict.get(r["CAMPAIGNID"], {}).get("budget", 1),
            "CLICK": r["PP_CLK"],
            "COST": r["PP_COST"],
            "PREV_CLICK": r["PP_PREV_CLK"],
            "PREV_CONV": r["PP_PREV_CONV"],
            "CONV": r["PP_CONV"],
            "SPENTBUDGET": r["SPENTBUDGET"],
            "LONG_IMP": r["PP_LONG_IMP"],
            "SHORT_IMP": r["PP_SHORT_IMP"],
            "IMP_INC": r["PP_IMP_INC"],
        }
        for r in sql_data
        if r["DEEPPPSWITCH"]
    ]
    ros_data = [
        {
            "CAMPAIGNID": r["CAMPAIGNID"],
            "CURR_CTR": r["ROS_CURR_CTR"],
            "DELTA_CTR": r["ROS_DELTA_CTR"],
            "CPC": r["ROS_CPC"],
            "BIDLOWER": r["ROSBIDLOWER"],
            "BIDUPPER": r["ROSBIDUPPER"],
            "CVR": r["ROS_CVR"],
            "CLICK_STEP": r["ROSCLICKSTEP"],
            "CVRFLAG": r["CVRFLAG"],
            "UNITS": r["UNITS"],
            "COMBINEDDEEPQSWITCH": r["COMBINEDDEEPQSWITCH"],
            "WEIGHTFLAG": r["WEIGHTFLAG"],
            "CTRWEIGHT": r["CTRWEIGHT"],
            "CONVWEIGHT": r["CONVWEIGHT"],
            "CVRWEIGHT": r["CVRWEIGHT"],
            "CPCWEIGHT": r["CPCWEIGHT"],
            "BUDGETWEIGHT": r["BUDGETWEIGHT"],
            "CURR_CPC": r["ROS_CURR_CPC"],
            "PREV_CPC": r["ROS_PREV_CPC"],
            "BASEBID": bid_dict.get(r["CAMPAIGNID"], 0),
            "BOOST": campaign_detail_dict.get(r["CAMPAIGNID"], {}).get("ROS", 0),
            "TOTALBID": get_totalbid_from_boost(
                bid_dict.get(r["CAMPAIGNID"], 0),
                campaign_detail_dict.get(r["CAMPAIGNID"], {}).get("ROS", 0),
            ),
            "STRATEGY": campaign_detail_dict.get(r["CAMPAIGNID"], {}).get(
                "strategy", "LEGACY_FOR_SALES"
            ),
            "BUDGET": campaign_detail_dict.get(r["CAMPAIGNID"], {}).get("budget", 1),
            "CLICK": r["ROS_CLK"],
            "COST": r["ROS_COST"],
            "PREV_CLICK": r["ROS_PREV_CLK"],
            "PREV_CONV": r["ROS_PREV_CONV"],
            "CONV": r["ROS_CONV"],
            "SPENTBUDGET": r["SPENTBUDGET"],
            "LONG_IMP": r["ROS_LONG_IMP"],
            "SHORT_IMP": r["ROS_SHORT_IMP"],
            "IMP_INC": r["ROS_IMP_INC"],
        }
        for r in sql_data
        if r["DEEPROSSWITCH"]
    ]

    tos_update_dict, tos_log, tos_reward, tos_reward_dist, tos_loss_dict = (
        run_deep_q_for_sp_spt(
            data=tos_data,
            campaignType="SPT",
            placement="TOS",
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            boltz_switch=boltz_switch,
            t_decay=t_decay,
        )
    )
    pp_update_dict, pp_log, pp_reward, pp_reward_dist, pp_loss_dict = (
        run_deep_q_for_sp_spt(
            data=pp_data,
            campaignType="SPT",
            placement="PP",
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            boltz_switch=boltz_switch,
            t_decay=t_decay,
        )
    )
    ros_update_dict, ros_log, ros_reward, ros_reward_dist, ros_loss_dict = (
        run_deep_q_for_sp_spt(
            data=ros_data,
            campaignType="SPT",
            placement="ROS",
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            boltz_switch=boltz_switch,
            t_decay=t_decay,
        )
    )

    spt_reward = None
    reward_list = []
    if tos_reward is not None:
        reward_list.append(tos_reward)
    if pp_reward is not None:
        reward_list.append(pp_reward)
    if ros_reward is not None:
        reward_list.append(ros_reward)

    if reward_list:
        spt_reward = sum(reward_list) / len(reward_list)

    result_dict = defaultdict(dict)
    for d in (tos_update_dict, pp_update_dict, ros_update_dict):
        for outer_key, inner_dict in d.items():
            result_dict[outer_key].update(inner_dict)

    final_boost_update_dict = dict(result_dict)
    boost_update_list = create_json_for_boost_update(final_boost_update_dict)

    if boost_update_list:
        sp.UPDATE_CAMPAIGNS_SP(boost_update_list)

    with open("spt_deepq.txt", "a") as f:
        f.write(
            pst_time_str()
            + "\n"
            + str(boost_update_list)
            + "\n"
            + f"TOS: {tos_log}"
            + "\n"
            + f"PP: {pp_log}"
            + "\n"
            + f"ROS: {ros_log}"
            + "\n"
            + f"TOS_loss: {tos_loss_dict}"
            + "\n"
            + f"PP_loss: {pp_loss_dict}"
            + "\n"
            + f"ROS_loss: {ros_loss_dict}"
            + "\n\n"
        )

    return spt_reward, add_distributions(
        tos_reward_dist, pp_reward_dist, ros_reward_dist
    )


def deep_q_videos_placement(interval: int, monitor: bool):
    sql_data = qcm.fetch(
        f"""
--SV PLACEMENT_TYPE
DECLARE @NOW DATETIME = '{pst_time_str()}',
	@INTERVAL INT = {interval},
	@MONITOR BIT = {int(monitor)};
DECLARE @STARTOFHOUR DATETIME2 = CAST(FORMAT(@NOW, 'yyyy-MM-dd HH:00') AS DATETIME2);
DECLARE @CVRDAYCOUNT INT = (SELECT MAX(CVRDAYCOUNT) FROM MODELING_DB.DBO.Q_MASTER WITH (NOLOCK)
    WHERE (DEEPPPSWITCH = 1 OR DEEPTOSSWITCH = 1 OR DEEPROSSWITCH = 1) AND CAMPAIGNTYPE = 'SV');
DECLARE @IMPVOLUMEHOUR INT = (SELECT MAX(IMPVOLUMEHOUR) FROM MODELING_DB.DBO.Q_MASTER WITH (NOLOCK)
    WHERE (DEEPTOSSWITCH = 1 OR DEEPPPSWITCH = 1 OR DEEPROSSWITCH = 1) AND CAMPAIGNTYPE IN ('SV'));
DECLARE @SHORTIMPHOUR INT = (SELECT MAX(SHORTIMPHOUR) FROM MODELING_DB.DBO.Q_MASTER WITH (NOLOCK)
    WHERE (DEEPTOSSWITCH = 1 OR DEEPPPSWITCH = 1 OR DEEPROSSWITCH = 1) AND CAMPAIGNTYPE IN ('SV'));
DECLARE @ITERATIONCOUNT INT = (SELECT MAX(ITERATIONCOUNT) FROM MODELING_DB.DBO.Q_MASTER WITH (NOLOCK)
    WHERE (DEEPTOSSWITCH = 1 OR DEEPPPSWITCH = 1 OR DEEPROSSWITCH = 1) AND CAMPAIGNTYPE IN ('SV'));
DECLARE @IMPRESSIONTHRESHOLD INT = (SELECT MAX(IMPRESSIONTHRESHOLD) FROM MODELING_DB.DBO.Q_MASTER WITH (NOLOCK)
    WHERE (DEEPTOSSWITCH = 1 OR DEEPPPSWITCH = 1 OR DEEPROSSWITCH = 1) AND CAMPAIGNTYPE IN ('SV'));
DECLARE @HOURLYTARGETIMP INT = (SELECT MAX(HOURLYTARGETIMP) FROM MODELING_DB.DBO.Q_MASTER WITH (NOLOCK)
    WHERE (DEEPTOSSWITCH = 1 OR DEEPPPSWITCH = 1 OR DEEPROSSWITCH = 1) AND CAMPAIGNTYPE IN ('SV'));

WITH MAIN_TAB AS (
        SELECT * FROM MODELING_DB.DBO.Q_MASTER WITH (NOLOCK)
        WHERE (DEEPPPSWITCH = 1 OR DEEPTOSSWITCH = 1 OR DEEPROSSWITCH = 1) AND CAMPAIGNTYPE = 'SV'),
TIME_TAB AS (
	SELECT CAMPAIGN_ID, PLACEMENT_TYPE, MAX( DATATIME) MAXTIME
	FROM [AMAZON_ANALYTICS].[DBO].AS_SB_TRAFFIC WITH (NOLOCK)
	WHERE CAST(TIME_WINDOW_START AS DATE) = CAST(@NOW AS DATE)
	AND DATATIME <= @NOW AND CAMPAIGN_ID IN (SELECT CAMPAIGNID FROM MAIN_TAB) GROUP BY campaign_id, placement_type),
SQS_TAB AS (
	SELECT CAMPAIGN_ID, PLACEMENT_TYPE, CLICKS, IMPRESSIONS, COST, DATATIME
	FROM [AMAZON_ANALYTICS].[DBO].AS_SB_TRAFFIC WITH (NOLOCK)
	WHERE CAST(TIME_WINDOW_START AS DATE) = CAST(@NOW AS DATE)
	AND DATATIME <= @NOW AND CAMPAIGN_ID IN (SELECT CAMPAIGNID FROM MAIN_TAB)),
SQS_TAB2 AS (
	SELECT A.CAMPAIGN_ID, A.PLACEMENT_TYPE, A.CLICKS, A.IMPRESSIONS, A.COST, A.DATATIME,TIME_TAB.MAXTIME
	FROM [AMAZON_ANALYTICS].[DBO].AS_SB_TRAFFIC A WITH (NOLOCK)
	JOIN TIME_TAB ON A.CAMPAIGN_ID = TIME_TAB.CAMPAIGN_ID AND A.placement_type = TIME_TAB.placement_type
	WHERE CAST(TIME_WINDOW_START AS DATE) = CAST(@NOW AS DATE)
		AND DATATIME <= @NOW AND A.CAMPAIGN_ID IN (SELECT CAMPAIGNID FROM MAIN_TAB)),
CURR_TAB AS (
	SELECT CAMPAIGN_ID, PLACEMENT_TYPE, SUM(CLICKS) CLK0, SUM(IMPRESSIONS) IMP0, SUM(CAST(COST AS FLOAT)) COST0,
		CASE WHEN SUM(CLICKS) = 0 THEN 0
		ELSE CAST(SUM(COST) AS FLOAT)/SUM(CLICKS) END AS CPC,
        CASE WHEN SUM(IMPRESSIONS) = 0 THEN 0
            WHEN SUM(IMPRESSIONS) > 0 AND SUM(CLICKS) <= 0 THEN (1.0/SUM(IMPRESSIONS) - 1)
        ELSE CAST(SUM(CLICKS) AS FLOAT)/SUM(IMPRESSIONS) END AS CTR0
        FROM SQS_TAB GROUP BY CAMPAIGN_ID, PLACEMENT_TYPE),
PREV_TAB AS (
        SELECT CAMPAIGN_ID, PLACEMENT_TYPE, SUM(CLICKS) CLK1, SUM(IMPRESSIONS) IMP1,
		CASE WHEN SUM(CLICKS) = 0 THEN 0
		ELSE CAST(SUM(COST) AS FLOAT)/SUM(CLICKS) END AS CPC1,
            CASE WHEN SUM(IMPRESSIONS) = 0 THEN 0 WHEN SUM(IMPRESSIONS) > 0 AND SUM(CLICKS) <= 0 THEN (1.0/SUM(IMPRESSIONS) - 1)
        ELSE CAST(SUM(CLICKS) AS FLOAT)/SUM(IMPRESSIONS) END AS CTR1
        FROM SQS_TAB WHERE DATATIME <= DATEADD(MINUTE, -@INTERVAL, @NOW)
        GROUP BY CAMPAIGN_ID, PLACEMENT_TYPE),
BAL_IMP_TAB AS(
	SELECT CAMPAIGN_ID, PLACEMENT_TYPE PLACEMENT, 100*(MAX(NOW_IMP) - MAX(BAL_ITER_IMP))/(MAX(BAL_ITER_IMP)+1) IMP_INC FROM(
		SELECT CAMPAIGN_ID, PLACEMENT_TYPE,  
		CASE WHEN @HOURLYTARGETIMP = -1 
				THEN CAST(SUM(IMPRESSIONS)AS FLOAT)/@ITERATIONCOUNT ELSE @HOURLYTARGETIMP END BAL_ITER_IMP, 0 NOW_IMP
		FROM SQS_TAB WHERE IMPRESSIONS >= 0 AND DATATIME BETWEEN DATEADD(MINUTE, -60*@ITERATIONCOUNT, @NOW) AND DATEADD(MINUTE, -60, @NOW)
		GROUP BY CAMPAIGN_ID, PLACEMENT_TYPE
		UNION
		SELECT CAMPAIGN_ID, PLACEMENT_TYPE, 0 BAL_ITER_IMP, CAST(SUM(IMPRESSIONS)AS FLOAT) NOW_IMP
		FROM SQS_TAB2 WHERE IMPRESSIONS >= 0 
		AND DATATIME BETWEEN DATEADD(MINUTE, -60, (CASE WHEN @HOURLYTARGETIMP = -1 THEN MAXTIME ELSE @NOW END)) 
		AND (CASE WHEN @HOURLYTARGETIMP = -1 THEN MAXTIME ELSE @NOW END)
		GROUP BY CAMPAIGN_ID, PLACEMENT_TYPE)T
        GROUP BY CAMPAIGN_ID, PLACEMENT_TYPE),
TOTAL_CLICK_TAB AS (
		SELECT CAMPAIGN_ID, PLACEMENT_TYPE, SUM(CLICKS) AS CLICKS FROM [AMAZON_ANALYTICS].[DBO].AS_SB_TRAFFIC WITH (NOLOCK)
		WHERE CAST(TIME_WINDOW_START AS DATETIME2) >= DATEADD(HOUR, -@CVRDAYCOUNT, @NOW) AND DATATIME <= @NOW
			AND CAMPAIGN_ID IN (SELECT CAMPAIGNID FROM MAIN_TAB)
		GROUP BY CAMPAIGN_ID, PLACEMENT_TYPE),
LONG_IMP_TAB AS (
	SELECT CAMPAIGN_ID, PLACEMENT, CAST(IMP AS FLOAT) LONG_IMP
	FROM (SELECT CAMPAIGN_ID, PLACEMENT_TYPE PLACEMENT, SUM(IMPRESSIONS) AS IMP FROM [AMAZON_ANALYTICS].[DBO].AS_SB_TRAFFIC WITH (NOLOCK)
		WHERE DATATIME BETWEEN DATEADD(HOUR, -@IMPVOLUMEHOUR, @NOW) AND @NOW
		GROUP BY CAMPAIGN_ID, PLACEMENT_TYPE)T),
SHORT_IMP_TAB AS (
	SELECT CAMPAIGN_ID, PLACEMENT, CAST(IMP AS FLOAT) SHORT_IMP
	FROM (SELECT CAMPAIGN_ID, PLACEMENT_TYPE PLACEMENT, SUM(IMPRESSIONS) AS IMP FROM [AMAZON_ANALYTICS].[DBO].AS_SB_TRAFFIC WITH (NOLOCK)
		WHERE DATATIME BETWEEN DATEADD(HOUR, -@SHORTIMPHOUR, @NOW) AND @NOW
		GROUP BY CAMPAIGN_ID, PLACEMENT_TYPE)T),
TOTAL_CONV_TAB AS (
		SELECT CAMPAIGN_ID, PLACEMENT_TYPE, SUM(ATTRIBUTED_CONVERSIONS_14D) AS CONVERSION FROM [AMAZON_ANALYTICS].[DBO].AS_SB_CONVERSION WITH (NOLOCK)
		WHERE CAST(TIME_WINDOW_START AS DATETIME2) >= DATEADD(HOUR, -@CVRDAYCOUNT, @NOW) AND DATATIME <= @NOW
			AND CAMPAIGN_ID IN (SELECT CAMPAIGNID FROM MAIN_TAB)
		GROUP BY CAMPAIGN_ID, PLACEMENT_TYPE),
CONV_TAB AS (
		SELECT SUM(CONVERSION) AS UNITS
		FROM (
			SELECT ISNULL(ORDERCOUNT - LAG(ORDERCOUNT) OVER (PARTITION BY CAST(TIMESTAMP AS DATE) ORDER BY TIMESTAMP), 0) AS CONVERSION
			FROM [AMAZON_MARKETING].[DBO].[AMZN_SALES_DATA] WITH (NOLOCK)
			WHERE TIMESTAMP >= DATEADD(HOUR, -1, @STARTOFHOUR)
			  AND TIMESTAMP <= @NOW
		) AS TAB),
CVR_TAB AS (
	SELECT TCLK.CAMPAIGN_ID, TCLK.PLACEMENT_TYPE, TCLK.CLICKS, TCV.CONVERSION,
		CASE WHEN TCV.CONVERSION>0 THEN CAST (TCLK.CLICKS AS FLOAT) / TCV.CONVERSION ELSE 0 END CVR
	FROM TOTAL_CLICK_TAB TCLK LEFT JOIN TOTAL_CONV_TAB TCV
		ON TCLK.CAMPAIGN_ID = TCV.CAMPAIGN_ID AND TCLK.PLACEMENT_TYPE = TCV.PLACEMENT_TYPE),
LATEST_CPC_TAB AS (
	SELECT CAMPAIGN_ID, PLACEMENT_TYPE, MAX(DATATIME) LATEST
	FROM SQS_TAB WHERE COST > 0 GROUP BY CAMPAIGN_ID, PLACEMENT_TYPE),
CPC_CURR_TAB AS(
	SELECT ST.CAMPAIGN_ID, ST.PLACEMENT_TYPE, SUM(CAST(ST.COST AS FLOAT)) / SUM(ST.CLICKS) AS CPC FROM
	SQS_TAB ST JOIN LATEST_CPC_TAB LCT ON ST.CAMPAIGN_ID = LCT.CAMPAIGN_ID AND ST.PLACEMENT_TYPE = LCT.PLACEMENT_TYPE AND ST.DATATIME = LCT.LATEST AND ST.CLICKS > 0
	GROUP BY ST.CAMPAIGN_ID, ST.PLACEMENT_TYPE),
CPC_PREV_TAB AS(
	SELECT ST.CAMPAIGN_ID, ST.PLACEMENT_TYPE, SUM(CAST(ST.COST AS FLOAT)) / SUM(ST.CLICKS) AS CPC FROM
	SQS_TAB ST JOIN(
		SELECT ST.CAMPAIGN_ID, ST.PLACEMENT_TYPE, MAX(ST.DATATIME) PREV
		FROM SQS_TAB ST JOIN LATEST_CPC_TAB LCT
			ON ST.CAMPAIGN_ID=LCT.CAMPAIGN_ID AND ST.PLACEMENT_TYPE=LCT.PLACEMENT_TYPE AND ST.DATATIME < LCT.LATEST AND ST.CLICKS > 0
		GROUP BY ST.CAMPAIGN_ID, ST.PLACEMENT_TYPE
	) PCT ON ST.CAMPAIGN_ID = PCT.CAMPAIGN_ID AND ST.PLACEMENT_TYPE = PCT.PLACEMENT_TYPE AND ST.DATATIME = PCT.PREV AND ST.CLICKS > 0
	GROUP BY ST.CAMPAIGN_ID, ST.PLACEMENT_TYPE),
COST_TAB AS (
	SELECT CAMPAIGN_ID, SUM(CAST(COST AS FLOAT)) COST FROM SQS_TAB GROUP BY CAMPAIGN_ID),
PLCMT_CONV_TAB AS (
	SELECT CAMPAIGN_ID, PLACEMENT_TYPE, SUM(ATTRIBUTED_CONVERSIONS_14D) AS CONV FROM [AMAZON_ANALYTICS].[DBO].AS_SB_CONVERSION WITH (NOLOCK)
		WHERE CAST(TIME_WINDOW_START AS DATE) >= CAST(@NOW AS DATE) AND DATATIME <= @NOW
			AND CAMPAIGN_ID IN (SELECT CAMPAIGNID FROM MAIN_TAB)
	GROUP BY CAMPAIGN_ID, PLACEMENT_TYPE),
PREV_PLCMT_CONV_TAB AS (
	SELECT CAMPAIGN_ID, PLACEMENT_TYPE, SUM(ATTRIBUTED_CONVERSIONS_14D) AS PREV_CONV FROM [AMAZON_ANALYTICS].[DBO].AS_SB_CONVERSION WITH (NOLOCK)
		WHERE CAST(TIME_WINDOW_START AS DATE) >= CAST(@NOW AS DATE) AND DATATIME <=  DATEADD(MINUTE, -@INTERVAL, @NOW)
			AND CAMPAIGN_ID IN (SELECT CAMPAIGNID FROM MAIN_TAB)
	GROUP BY CAMPAIGN_ID, PLACEMENT_TYPE),
JOINED_TAB AS (
	SELECT CT.CAMPAIGN_ID, CT.PLACEMENT_TYPE, CT.CLK0, CT.IMP0, CT.CTR0, CT.CPC, CAST(CT.COST0 AS FLOAT) COST0, LIT.LONG_IMP, SIT.SHORT_IMP,
		PT.CLK1, PT.IMP1, PT.CTR1, PT.CPC1, CVT.CVR, CPC_C.CPC AS CURR_CPC, CPC_P.CPC AS PREV_CPC, PCT.CONV, PVCT.PREV_CONV,
		CASE WHEN @HOURLYTARGETIMP != -1 THEN
				CASE WHEN ISNULL([BIT].IMP_INC, 0) < @IMPRESSIONTHRESHOLD THEN 0 ELSE ISNULL([BIT].IMP_INC, 0) END 
			ELSE ISNULL([BIT].IMP_INC, 0) END IMP_INC,
		CASE WHEN ISNULL(CT.CLK0, 0) - ISNULL(PT.CLK1, 0) <= 0 OR ISNULL(CT.IMP0, 0) - ISNULL(PT.IMP1, 0) <= 0 THEN 0
			ELSE CAST(ISNULL(CT.CLK0, 0) - ISNULL(PT.CLK1, 0) AS FLOAT) / (ISNULL(CT.IMP0, 0) - ISNULL(PT.IMP1, 0)) END CTR_IMPRV
	FROM CURR_TAB CT LEFT JOIN PREV_TAB PT ON CT.CAMPAIGN_ID=PT.CAMPAIGN_ID AND CT.PLACEMENT_TYPE=PT.PLACEMENT_TYPE
		LEFT JOIN CVR_TAB CVT ON CT.CAMPAIGN_ID=CVT.CAMPAIGN_ID AND CT.PLACEMENT_TYPE=CVT.PLACEMENT_TYPE
        LEFT JOIN CPC_CURR_TAB CPC_C ON CT.CAMPAIGN_ID=CPC_C.CAMPAIGN_ID AND CT.PLACEMENT_TYPE=CPC_C.PLACEMENT_TYPE
		LEFT JOIN CPC_PREV_TAB CPC_P ON CT.CAMPAIGN_ID=CPC_P.CAMPAIGN_ID AND CT.PLACEMENT_TYPE=CPC_P.PLACEMENT_TYPE
        LEFT JOIN PREV_PLCMT_CONV_TAB PVCT ON CT.CAMPAIGN_ID=PVCT.CAMPAIGN_ID AND CT.PLACEMENT_TYPE=PVCT.PLACEMENT_TYPE
		LEFT JOIN LONG_IMP_TAB LIT ON CT.CAMPAIGN_ID=LIT.CAMPAIGN_ID AND CT.PLACEMENT_TYPE=LIT.PLACEMENT
        LEFT JOIN SHORT_IMP_TAB SIT ON CT.CAMPAIGN_ID=SIT.CAMPAIGN_ID AND CT.PLACEMENT_TYPE=SIT.PLACEMENT
		LEFT JOIN PLCMT_CONV_TAB PCT ON CT.CAMPAIGN_ID=PCT.CAMPAIGN_ID AND CT.PLACEMENT_TYPE=PCT.PLACEMENT_TYPE
		LEFT JOIN BAL_IMP_TAB [BIT] ON CT.CAMPAIGN_ID=[BIT].CAMPAIGN_ID AND CT.PLACEMENT_TYPE = [BIT].PLACEMENT),
JOINED_TAB1 AS (
	SELECT *, CASE WHEN SUM(CTR_IMPRV) OVER () <= 0 THEN 0 ELSE CTR_IMPRV / SUM(CTR_IMPRV) OVER () END CTR_SHARE
	FROM JOINED_TAB),
MERGED_TAB AS(
	SELECT MT.CAMPAIGNID,MT.GAMMA, MT.LR, MT.EPSTART, MT.DEEPPPSWITCH, MT.DEEPTOSSWITCH, MT.DEEPROSSWITCH, MT.COMBINEDDEEPQSWITCH,
		MT.PPBIDLOWER, MT.PPBIDUPPER,MT.TOSBIDLOWER, MT.TOSBIDUPPER, MT.ROSBIDLOWER, MT.ROSBIDUPPER, MT.CVRFLAG, MT.PPCLICKSTEP, MT.TOSCLICKSTEP, MT.ROSCLICKSTEP, MT.DYNAMICEXPLORE,
		ISNULL(PP_T.CVR,0) PP_CVR, ISNULL(TOS_T.CVR,0) TOS_CVR, ISNULL(ROS_T.CVR,0) ROS_CVR, 
        MT.WEIGHTFLAG, MT.CTRWEIGHT, MT.CONVWEIGHT, MT.CVRWEIGHT, MT.CPCWEIGHT, MT.BUDGETWEIGHT, MT.ONLYDROP, MT.DROPIMPROVED,
        MT.BOLTZFLAG, MT.TDECAY,
		ISNULL(PP_T.CTR0,0) PP_CURR_CTR, ISNULL(TOS_T.CTR0,0) TOS_CURR_CTR, ISNULL(ROS_T.CTR0,0) ROS_CURR_CTR, 
		ISNULL(ROUND(PP_T.CTR0 - ISNULL(PP_T.CTR1,0),4),0) PP_DELTA_CTR, ISNULL(ROUND(TOS_T.CTR0 - ISNULL(TOS_T.CTR1,0),4),0) TOS_DELTA_CTR, ISNULL(ROUND(ROS_T.CTR0 - ISNULL(ROS_T.CTR1,0),4),0) ROS_DELTA_CTR,
		ISNULL(PP_T.CPC,0) PP_CPC, ISNULL(TOS_T.CPC,0) TOS_CPC, ISNULL(ROS_T.CPC,0) ROS_CPC, 
		ISNULL(PP_T.CPC1,0) PP_CPC1, ISNULL(TOS_T.CPC1,0) TOS_CPC1, ISNULL(ROS_T.CPC1,0) ROS_CPC1,
        ISNULL(PP_T.CURR_CPC,0) PP_CURR_CPC, ISNULL(TOS_T.CURR_CPC,0) TOS_CURR_CPC, ISNULL(ROS_T.CURR_CPC,0) ROS_CURR_CPC, 
		ISNULL(PP_T.PREV_CPC,0) PP_PREV_CPC, ISNULL(TOS_T.PREV_CPC,0) TOS_PREV_CPC, ISNULL(ROS_T.PREV_CPC,0) ROS_PREV_CPC, 
		ISNULL(PP_T.CLK1,0) PP_PREV_CLK, ISNULL(TOS_T.CLK1,0) TOS_PREV_CLK, ISNULL(ROS_T.CLK1,0) ROS_PREV_CLK,
		ISNULL(PP_T.CLK0,0) PP_CLK, ISNULL(TOS_T.CLK0,0) TOS_CLK, ISNULL(ROS_T.CLK0,0) ROS_CLK,
		ISNULL(PP_T.COST0,0) PP_COST, ISNULL(TOS_T.COST0,0) TOS_COST, ISNULL(ROS_T.COST0,0) ROS_COST,
		ISNULL(PP_T.PREV_CONV,0) PP_PREV_CONV, ISNULL(TOS_T.PREV_CONV,0) TOS_PREV_CONV, ISNULL(ROS_T.PREV_CONV,0) ROS_PREV_CONV,
		ISNULL(PP_T.CONV,0) PP_CONV, ISNULL(TOS_T.CONV,0) TOS_CONV, ISNULL(ROS_T.CONV,0) ROS_CONV,
		ISNULL(PP_T.CTR_SHARE, 0) PP_CTR_SHR, ISNULL(TOS_T.CTR_SHARE, 0) TOS_CTR_SHR, ISNULL(ROS_T.CTR_SHARE, 0) ROS_CTR_SHR,
		ISNULL(PP_T.LONG_IMP, 0) PP_LONG_IMP,  ISNULL(TOS_T.LONG_IMP, 0) TOS_LONG_IMP, ISNULL(ROS_T.LONG_IMP, 0) ROS_LONG_IMP,
		ISNULL(PP_T.SHORT_IMP, 0) PP_SHORT_IMP,  ISNULL(TOS_T.SHORT_IMP, 0) TOS_SHORT_IMP, ISNULL(ROS_T.SHORT_IMP, 0) ROS_SHORT_IMP,
		ISNULL(PP_T.IMP_INC,0)PP_IMP_INC, ISNULL(TOS_T.IMP_INC,0)TOS_IMP_INC, ISNULL(ROS_T.IMP_INC,0)ROS_IMP_INC
	FROM MAIN_TAB MT LEFT JOIN (
		SELECT * FROM JOINED_TAB1 WHERE PLACEMENT_TYPE='DETAIL PAGE ON-AMAZON') PP_T ON MT.CAMPAIGNID=PP_T.CAMPAIGN_ID
		LEFT JOIN (SELECT * FROM JOINED_TAB1 WHERE PLACEMENT_TYPE='OTHER ON-AMAZON')ROS_T ON MT.CAMPAIGNID=ROS_T.CAMPAIGN_ID
		LEFT JOIN (SELECT * FROM JOINED_TAB1 WHERE PLACEMENT_TYPE='TOP OF SEARCH ON-AMAZON')TOS_T ON MT.CAMPAIGNID=TOS_T.CAMPAIGN_ID
),
MERGED_WITH_CONV AS (
	SELECT MT.*, ISNULL(CVT.UNITS,0) AS UNITS, ISNULL(CST.COST,0) SPENTBUDGET
	FROM MERGED_TAB MT LEFT JOIN COST_TAB CST ON MT.CAMPAIGNID = CST.CAMPAIGN_ID
	LEFT JOIN CONV_TAB CVT ON 1=1
)
SELECT * 
FROM MERGED_WITH_CONV
WHERE @MONITOR = 0 OR (ROS_DELTA_CTR <= 0 OR PP_DELTA_CTR <= 0)
    """
    )
    if not sql_data:
        with open("sv_boost_deepq.txt", "a") as f:
            f.write(pst_time_str() + "\n" + "No Data!" + "\n\n")
            return None, [[0, 0, 0] for _ in range(100)]

    dynamic_explore = sql_data[0]["DYNAMICEXPLORE"]
    learning_rate = sql_data[0]["LR"]
    gamma = sql_data[0]["GAMMA"]
    epsilon = (
        sql_data[0]["EPSTART"]
        if not dynamic_explore
        else round(
            (min(0.99, sql_data[0]["EPSTART"]) ** (minute_from_last_reset / 60)), 2
        )
    )
    boltz_switch = sql_data[0]["BOLTZFLAG"]
    t_decay = sql_data[0]["TDECAY"]

    campaign_detail_dict = get_sv_campaign_detail([r["CAMPAIGNID"] for r in sql_data])
    bid_dict = get_video_avgbids([r["CAMPAIGNID"] for r in sql_data])

    pp_data = [
        {
            "CAMPAIGNID": r["CAMPAIGNID"],
            "CURR_CTR": r["PP_CURR_CTR"],
            "DELTA_CTR": r["PP_DELTA_CTR"],
            "CPC": r["PP_CPC"],
            "BIDLOWER": r["PPBIDLOWER"],
            "BIDUPPER": r["PPBIDUPPER"],
            "CVR": r["PP_CVR"],
            "CLICK_STEP": r["PPCLICKSTEP"],
            "CVRFLAG": r["CVRFLAG"],
            "UNITS": r["UNITS"],
            "COMBINEDDEEPQSWITCH": r["COMBINEDDEEPQSWITCH"],
            "WEIGHTFLAG": r["WEIGHTFLAG"],
            "CTRWEIGHT": r["CTRWEIGHT"],
            "CONVWEIGHT": r["CONVWEIGHT"],
            "CVRWEIGHT": r["CVRWEIGHT"],
            "CPCWEIGHT": r["CPCWEIGHT"],
            "BUDGETWEIGHT": r["BUDGETWEIGHT"],
            "CURR_CPC": r["PP_CURR_CPC"],
            "PREV_CPC": r["PP_PREV_CPC"],
            "BASEBID": bid_dict.get(r["CAMPAIGNID"], 0),
            "BOOST": campaign_detail_dict.get(r["CAMPAIGNID"], {}).get("PP", 0),
            "TOTALBID": get_totalbid_from_boost(
                bid_dict.get(r["CAMPAIGNID"], 0),
                campaign_detail_dict.get(r["CAMPAIGNID"], {}).get("PP", 0),
            ),
            "STRATEGY": campaign_detail_dict.get(r["CAMPAIGNID"], {}).get(
                "strategy", "MAXIMIZE_IMMEDIATE_SALES"
            ),
            "BUDGET": campaign_detail_dict.get(r["CAMPAIGNID"], {}).get("budget", 1),
            "CLICK": r["PP_CLK"],
            "COST": r["PP_COST"],
            "PREV_CLICK": r["PP_PREV_CLK"],
            "PREV_CONV": r["PP_PREV_CONV"],
            "CONV": r["PP_CONV"],
            "SPENTBUDGET": r["SPENTBUDGET"],
            "CTR_SHR": r["PP_CTR_SHR"],
            "LONG_IMP": r["PP_LONG_IMP"],
            "SHORT_IMP": r["PP_SHORT_IMP"],
            "IMP_INC": r["PP_IMP_INC"],
        }
        for r in sql_data
        if r["DEEPPPSWITCH"]
    ]
    tos_data = [
        {
            "CAMPAIGNID": r["CAMPAIGNID"],
            "CURR_CTR": r["TOS_CURR_CTR"],
            "DELTA_CTR": r["TOS_DELTA_CTR"],
            "CPC": r["TOS_CPC"],
            "BIDLOWER": r["TOSBIDLOWER"],
            "BIDUPPER": r["TOSBIDUPPER"],
            "CVR": r["TOS_CVR"],
            "CLICK_STEP": r["TOSCLICKSTEP"],
            "CVRFLAG": r["CVRFLAG"],
            "UNITS": r["UNITS"],
            "COMBINEDDEEPQSWITCH": r["COMBINEDDEEPQSWITCH"],
            "WEIGHTFLAG": r["WEIGHTFLAG"],
            "CTRWEIGHT": r["CTRWEIGHT"],
            "CONVWEIGHT": r["CONVWEIGHT"],
            "CVRWEIGHT": r["CVRWEIGHT"],
            "CPCWEIGHT": r["CPCWEIGHT"],
            "BUDGETWEIGHT": r["BUDGETWEIGHT"],
            "CURR_CPC": r["TOS_CURR_CPC"],
            "PREV_CPC": r["TOS_PREV_CPC"],
            "BASEBID": bid_dict.get(r["CAMPAIGNID"], 0),
            "BOOST": campaign_detail_dict.get(r["CAMPAIGNID"], {}).get("TOS", 0),
            "TOTALBID": get_totalbid_from_boost(
                bid_dict.get(r["CAMPAIGNID"], 0),
                campaign_detail_dict.get(r["CAMPAIGNID"], {}).get("TOS", 0),
            ),
            "STRATEGY": campaign_detail_dict.get(r["CAMPAIGNID"], {}).get(
                "strategy", "MAXIMIZE_IMMEDIATE_SALES"
            ),
            "BUDGET": campaign_detail_dict.get(r["CAMPAIGNID"], {}).get("budget", 1),
            "CLICK": r["TOS_CLK"],
            "COST": r["TOS_COST"],
            "PREV_CLICK": r["TOS_PREV_CLK"],
            "PREV_CONV": r["TOS_PREV_CONV"],
            "CONV": r["TOS_CONV"],
            "SPENTBUDGET": r["SPENTBUDGET"],
            "CTR_SHR": r["TOS_CTR_SHR"],
            "LONG_IMP": r["TOS_LONG_IMP"],
            "SHORT_IMP": r["TOS_SHORT_IMP"],
            "IMP_INC": r["TOS_IMP_INC"],
        }
        for r in sql_data
        if r["DEEPTOSSWITCH"]
    ]

    ros_data = [
        {
            "CAMPAIGNID": r["CAMPAIGNID"],
            "CURR_CTR": r["ROS_CURR_CTR"],
            "DELTA_CTR": r["ROS_DELTA_CTR"],
            "CPC": r["ROS_CPC"],
            "BIDLOWER": r["ROSBIDLOWER"],
            "BIDUPPER": r["ROSBIDUPPER"],
            "CVR": r["ROS_CVR"],
            "CLICK_STEP": r["ROSCLICKSTEP"],
            "CVRFLAG": r["CVRFLAG"],
            "UNITS": r["UNITS"],
            "COMBINEDDEEPQSWITCH": r["COMBINEDDEEPQSWITCH"],
            "WEIGHTFLAG": r["WEIGHTFLAG"],
            "CTRWEIGHT": r["CTRWEIGHT"],
            "CONVWEIGHT": r["CONVWEIGHT"],
            "CVRWEIGHT": r["CVRWEIGHT"],
            "CPCWEIGHT": r["CPCWEIGHT"],
            "BUDGETWEIGHT": r["BUDGETWEIGHT"],
            "CURR_CPC": r["ROS_CURR_CPC"],
            "PREV_CPC": r["ROS_PREV_CPC"],
            "BASEBID": bid_dict.get(r["CAMPAIGNID"], 0),
            "BOOST": campaign_detail_dict.get(r["CAMPAIGNID"], {}).get("ROS", 0),
            "TOTALBID": get_totalbid_from_boost(
                bid_dict.get(r["CAMPAIGNID"], 0),
                campaign_detail_dict.get(r["CAMPAIGNID"], {}).get("ROS", 0),
            ),
            "STRATEGY": campaign_detail_dict.get(r["CAMPAIGNID"], {}).get(
                "strategy", "MAXIMIZE_IMMEDIATE_SALES"
            ),
            "BUDGET": campaign_detail_dict.get(r["CAMPAIGNID"], {}).get("budget", 1),
            "CLICK": r["ROS_CLK"],
            "COST": r["ROS_COST"],
            "PREV_CLICK": r["ROS_PREV_CLK"],
            "PREV_CONV": r["ROS_PREV_CONV"],
            "CONV": r["ROS_CONV"],
            "SPENTBUDGET": r["SPENTBUDGET"],
            "CTR_SHR": r["ROS_CTR_SHR"],
            "LONG_IMP": r["ROS_LONG_IMP"],
            "SHORT_IMP": r["ROS_SHORT_IMP"],
            "IMP_INC": r["ROS_IMP_INC"],
        }
        for r in sql_data
        if r["DEEPROSSWITCH"]
    ]

    pp_update_dict, pp_log, pp_reward, pp_reward_dist, pp_loss_dict = (
        run_deep_q_for_sp_spt(
            data=pp_data,
            campaignType="SV",
            placement="PP",
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            boltz_switch=boltz_switch,
            t_decay=t_decay,
        )
    )
    tos_update_dict, tos_log, tos_reward, tos_reward_dist, tos_loss_dict = (
        run_deep_q_for_sp_spt(
            data=tos_data,
            campaignType="SV",
            placement="TOS",
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            boltz_switch=boltz_switch,
            t_decay=t_decay,
        )
    )
    ros_update_dict, ros_log, ros_reward, ros_reward_dist, ros_loss_dict = (
        run_deep_q_for_sp_spt(
            data=ros_data,
            campaignType="SV",
            placement="ROS",
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            boltz_switch=boltz_switch,
            t_decay=t_decay,
        )
    )

    sv_boost_reward = None
    reward_list = []
    if pp_reward is not None:
        reward_list.append(pp_reward)
    if tos_reward is not None:
        reward_list.append(tos_reward)
    if ros_reward is not None:
        reward_list.append(ros_reward)

    if reward_list:
        sv_boost_reward = sum(reward_list) / len(reward_list)

    result_dict = defaultdict(dict)
    for d in (pp_update_dict, tos_update_dict, ros_update_dict):
        for outer_key, inner_dict in d.items():
            result_dict[outer_key].update(inner_dict)

    final_boost_update_dict = dict(result_dict)
    boost_update_list = create_json_for_boost_update_sv(final_boost_update_dict)

    if boost_update_list:
        sb.UPDATE_CAMPAIGNS_SB(boost_update_list)

    with open("sv_boost_deepq.txt", "a") as f:
        f.write(
            pst_time_str()
            + "\n"
            + str(boost_update_list)
            + "\n"
            + f"PP: {pp_log}"
            + "\n"
            + f"TOS: {tos_log}"
            + "\n"
            + f"ROS: {ros_log}"
            + "\n"
            + f"PP_loss: {pp_loss_dict}"
            + "\n"
            + f"TOS_loss: {tos_loss_dict}"
            + "\n"
            + f"ROS_loss: {ros_loss_dict}"
            + "\n\n"
        )

    return sv_boost_reward, add_distributions(
        pp_reward_dist, tos_reward_dist, ros_reward_dist
    )


while True:
    try:
        reward_list = []
        interval_dict = {
            r["CAMPAIGNTYPE"]: [r["INTERVAL"], r["MONITOR"]]
            for r in qcm.fetch(
                """ SELECT CAMPAIGNTYPE, MAX(INTERVAL) INTERVAL, CAST(MAX(CAST(MONITOR AS INT)) AS BIT) MONITOR
                FROM MODELING_DB.DBO.Q_MASTER GROUP BY CAMPAIGNTYPE """
            )
        }
        record = qcm.fetch(
            "select top 1 interval, epstart, dynamicexplore from MODELING_DB.DBO.Q_MASTER where combineddeepqswitch = 1"
        )[0]
        interval = record["interval"]
        ep_start = record["epstart"]
        dynamic_explore = record["dynamicexplore"]

        dm_res = qcm.fetch(
            f"""
            DECLARE @resetTime DATETIME = (
                    SELECT TOP 1 [timestamp]
                    FROM [MODELING_DB].[dbo].[deepq_reward_table]
                    WHERE avgReward = - 999
                    ORDER BY TIMESTAMP DESC
                    );
            DECLARE @currPst DATETIME = '{pst_time_str()}';
            SELECT ISNULL(DATEDIFF(HOUR, @resetTime, @currPst), 0) AS DM
            """
        )
        minute_from_last_reset = dm_res[0]["DM"]
        current_minute = pst_time().minute
        to_monitor = current_minute % 60 != 0
        sv_interval, sv_monitor = interval_dict["SV"]
        sp_interval, sp_monitor = interval_dict["SP"]
        spt_interval, spt_monitor = interval_dict["SPT"]
        sd_interval, sd_monitor = interval_dict["SD"]

        sv_reward_dist = [[0, 0, 0] for _ in range(100)]
        sp_reward_dist = [[0, 0, 0] for _ in range(100)]
        spt_reward_dist = [[0, 0, 0] for _ in range(100)]
        sd_reward_dist = [[0, 0, 0] for _ in range(100)]

        if current_minute % sv_interval == 0:
            try:
                sv_reward, sv_reward_dist = deep_q_videos(
                    interval=sv_interval, monitor=sv_monitor and to_monitor
                )
                if sv_reward is not None:
                    reward_list.append(sv_reward)
            except Exception as e:
                with open("deep_q_error_sv.txt", "a") as f:
                    f.write(pst_time_str() + "=>" + str(e) + "\n\n")
            try:
                sv_boost_reward, sv_boost_reward_dist = deep_q_videos_placement(
                    interval=sv_interval, monitor=sv_monitor and to_monitor
                )
                if sv_boost_reward is not None:
                    reward_list.append(sv_boost_reward)
            except Exception as e:
                with open("deep_q_error_sv_boost.txt", "a") as f:
                    f.write(pst_time_str() + "=>" + str(e) + "\n\n")
        if current_minute % sp_interval == 0:
            try:
                sp_reward, sp_reward_dist = deep_q_sp(
                    interval=sp_interval, monitor=sp_monitor and to_monitor
                )
                if sp_reward is not None:
                    reward_list.append(sp_reward)
            except Exception as e:
                with open("deep_q_error_sp.txt", "a") as f:
                    f.write(pst_time_str() + "=>" + str(e) + "\n\n")
            try:
                sp_bid_reward = deep_q_sp_bid(
                    interval=sp_interval, monitor=sp_monitor and to_monitor
                )
                if sp_bid_reward is not None:
                    reward_list.append(sp_bid_reward)
            except Exception as e:
                with open("deep_q_error_sp_bid.txt", "a") as f:
                    f.write(pst_time_str() + "=>" + str(e) + "\n\n")
        if current_minute % spt_interval == 0:
            try:
                spt_reward, spt_reward_dist = deep_q_spt(
                    interval=spt_interval, monitor=spt_monitor and to_monitor
                )
                if spt_reward is not None:
                    reward_list.append(spt_reward)
            except Exception as e:
                with open("deep_q_error_spt.txt", "a") as f:
                    f.write(pst_time_str() + "=>" + str(e) + "\n\n")
        if current_minute % sd_interval == 0:
            try:
                sd_reward, sd_reward_dist = deep_q_display(
                    interval=sd_interval, monitor=sd_monitor and to_monitor
                )
                if sd_reward is not None:
                    reward_list.append(sd_reward)
            except Exception as e:
                with open("deep_q_error_sd.txt", "a") as f:
                    f.write(pst_time_str() + "=>" + str(e) + "\n\n")

        all_reward_distribution = add_distributions(
            sv_reward_dist, sp_reward_dist, spt_reward_dist, sd_reward_dist
        )
        if any([any(i) for i in all_reward_distribution]):
            total_reward_dist, ctr_reward_dist, cvr_reward_dist = [], [], []
            for i in all_reward_distribution:
                total_reward_dist.append(i[0])
                ctr_reward_dist.append(i[1])
                cvr_reward_dist.append(i[2])
            qcm.insert_many(
                table_name="[modeling_db].[dbo].deepq_distributions",
                data=[
                    {
                        "rewardType": "total",
                        "distribution": str(total_reward_dist),
                        "timestamp": pst_time_str(),
                    },
                    {
                        "rewardType": "del_ctr",
                        "distribution": str(ctr_reward_dist),
                        "timestamp": pst_time_str(),
                    },
                    {
                        "rewardType": "del_cvr",
                        "distribution": str(cvr_reward_dist),
                        "timestamp": pst_time_str(),
                    },
                ],
            )

        if reward_list:
            reward_data = {
                "campType": "GLOBAL",
                "avgReward": sum(reward_list) / len(reward_list),
                "timestamp": pst_time_str(),
            }
            qcm.insert_many(table_name="deepq_reward_table", data=[reward_data])

        if current_minute % interval == 0:
            epsilon_data = {
                "campType": "GLOBAL",
                "epsilon": (
                    ep_start
                    if not dynamic_explore
                    else round(
                        (min(0.99, ep_start) ** (minute_from_last_reset / 60)), 2
                    )
                ),
                "timestamp": pst_time_str(),
            }
            qcm.insert_many(table_name="deepq_epsilon_table", data=[epsilon_data])

        now = pst_time()
        sleep_time = (
            datetime.strptime(
                datetime.strftime(now, "%Y-%m-%d %H:%M"), "%Y-%m-%d %H:%M"
            )
            + timedelta(minutes=1)
            - now
        ).total_seconds()
        time.sleep(sleep_time)

    except Exception as e:
        with open("deepq_central_error.txt", "a") as f:
            f.write(pst_time_str() + "\n:Error: =>" + str(e) + "\n\n")
