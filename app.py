# ============================================================
# COVID-19 Home Diagnostic AI
# Streamlit Application
# ============================================================

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import re

import numpy as np
import pandas as pd
import streamlit as st
import networkx as nx

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.inference import VariableElimination


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="COVID-19 Home Diagnostic AI",
    page_icon="🩺",
    layout="wide"
)

RANDOM_STATE = 42
TARGET = "PCR Test Positive"
HOME_TEST_VAR = "HOME_TEST_RESULT_PRE_PCR"


# ============================================================
# LOAD AND BUILD MODEL
# ============================================================

@st.cache_resource
def build_covid_model():

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    DATA_DIR = Path("data")

    df = pd.read_csv(
        DATA_DIR /
        "COVIDCARE_FORSUBMISSION_MIT_CLEANED_Phase_II_2021-12-03.csv"
    )

    dictionary = pd.read_csv(
        DATA_DIR /
        "COVIDCARE_survey_dictionary_v2_ForSubmission_MIT_Phase_II_2021-12-26.csv"
    )

    kb = pd.read_csv(
        DATA_DIR /
        "COVIDCARE_DEMI_knowledgebase_v4.csv"
    )

    df.columns = df.columns.str.strip()
    dictionary.columns = dictionary.columns.str.strip()
    kb.columns = kb.columns.str.strip()


    # --------------------------------------------------------
    # 2. Dictionary functions
    # --------------------------------------------------------

    dictionary["Variable Name"] = (
        dictionary["Variable Name"]
        .astype(str)
        .str.strip()
    )

    DICT_MAP = (
        dictionary
        .set_index("Variable Name")
        .to_dict("index")
    )

    def dict_value(var, field, default=""):

        row = DICT_MAP.get(var, {})

        value = str(
            row.get(field, default)
        ).strip()

        if value.lower() == "nan" or value == "":
            return default

        return value


    def description_of(var):

        return dict_value(
            var,
            "Description",
            var
        )


    def prompt_of(var):

        return dict_value(
            var,
            "Prompt",
            ""
        )


    def form_of(var):

        return dict_value(
            var,
            "Comments (Optional, Form Collection Name)",
            "Unknown"
        )


    def datatype_of(var):

        return dict_value(
            var,
            "Data Type",
            "Unknown"
        )


    # --------------------------------------------------------
    # 3. Short descriptions
    # --------------------------------------------------------

    def short_description(var):

        if var == HOME_TEST_VAR:
            return "At-home COVID test result"

        desc = description_of(var)
        prompt = prompt_of(var)

        text = prompt if prompt else desc

        if "?" in text:

            specific = (
                text
                .split("?")[-1]
                .strip()
            )

            if specific:
                return specific

        if ";" in text:

            specific = (
                text
                .split(";")[-1]
                .strip()
            )

            if specific:
                return specific

        return desc


    # Cleaner display labels
    LABEL_OVERRIDES = {

        "HOME_TEST_RESULT_PRE_PCR":
            "At-home COVID test",

        "30075-Other_household":
            "Number of household members",

        "30166-Prev_exposure-1":
            "Recent COVID exposure",

        "30103-conditionsrisks-13":
            "Mental health condition",

        "32136-vaccine_didyou":
            "Vaccination status",

        "32137-vaccine_avail":
            "Likelihood to receive vaccine",

        "30101-ace_medication":
            "ACE medication",

        "30158-Symtpom_Neuro-7":
            "Loss of smell",

        "30158-Symtpom_Neuro-8":
            "Loss of taste",

        "30172-COVID_vaccine_type":
            "COVID vaccine type"
    }


    def display_name(var):

        if var in LABEL_OVERRIDES:
            return LABEL_OVERRIDES[var]

        return short_description(var)


    # --------------------------------------------------------
    # 4. Classify home-available variables
    # --------------------------------------------------------

    POST_VISIT_FORMS = {

        "Follow-up Survey",

        "I Completed My In-Clinic COVID Test",

        "Confirm Your Shipping Address",

        "I Received My Testing Kit",

        "Consent"
    }


    PRIMARY_HOME_FORMS = {
        "About You",
        "Symptom Screening"
    }


    AT_HOME_TEST_FORM = (
        "At-home COVID Kit Instructions and Survey"
    )


    PRIOR_TEST_PATTERNS = [

        r"covid_result",
        r"covid_tested",
        r"covid_tst",
        r"covid_why",
        r"test_specify",
        r"test_date"
    ]


    ADMIN_PATTERNS = [

        r"submission",
        r"confirmation",
        r"internal.?id",
        r"_deid$",
        r"email",
        r"phone",
        r"address",
        r"shipping",
        r"cohort"
    ]


    def matches_any(text, patterns):

        text = str(text)

        return any(
            re.search(
                pattern,
                text,
                flags=re.I
            )
            for pattern in patterns
        )


    def classify_variable(var):

        form = form_of(var)
        dtype = datatype_of(var)

        combined = (
            f"{var} "
            f"{description_of(var)} "
            f"{prompt_of(var)}"
        ).lower()


        if var == TARGET:

            return (
                "Not available at home",
                False
            )


        if var == "PCR Test Date_DEID":

            return (
                "Not available at home",
                False
            )


        if form == AT_HOME_TEST_FORM:

            return (
                "Available at home",
                False
            )


        if form in POST_VISIT_FORMS:

            return (
                "Not available at home",
                False
            )


        if form == "Unknown":

            return (
                "Uncertain",
                False
            )


        if form in PRIMARY_HOME_FORMS:

            if dtype.lower() == "text":

                return (
                    "Available at home",
                    False
                )


            if matches_any(
                var,
                ADMIN_PATTERNS
            ):

                return (
                    "Not available at home",
                    False
                )


            if matches_any(
                var,
                PRIOR_TEST_PATTERNS
            ):

                return (
                    "Available at home",
                    False
                )


            if re.search(
                r"flu_(tst|why)",
                var,
                flags=re.I
            ):

                return (
                    "Available at home",
                    False
                )


            lab_terms = [

                "lab-administered",
                "pcr test",
                "antigen test",
                "antibody test"
            ]


            if any(
                term in combined
                for term in lab_terms
            ):

                return (
                    "Not available at home",
                    False
                )


            return (
                "Available at home",
                True
            )


        return (
            "Uncertain",
            False
        )


    classification_rows = []

    for var in df.columns:

        status, use = (
            classify_variable(var)
        )

        classification_rows.append({

            "Variable": var,

            "Home_Classification":
                status,

            "Primary_Model_Use":
                use
        })


    classification = pd.DataFrame(
        classification_rows
    )


    # --------------------------------------------------------
    # 5. Derive home-test result
    # --------------------------------------------------------

    HOME_TEST_SPECS = [

        {
            "date":
                "32007-datestart_DEID",

            "positive":
                "30766-pinkblue_confirm",

            "negative":
                "30769-blue_nopink_confirm",

            "invalid":
                "30772-noblue_confirm"
        },

        {
            "date":
                "32320-datestart_2_DEID",

            "positive":
                "32356-pinkblue_confirm_2",

            "negative":
                "32359-blue_nopink_confirm_2",

            "invalid":
                "32362-noblue_confirm_2"
        }
    ]


    PCR_DATE = "PCR Test Date_DEID"


    def derive_home_test_result(row):

        pcr_date = row.get(
            PCR_DATE,
            np.nan
        )

        statuses = []


        for spec in HOME_TEST_SPECS:

            home_date = row.get(
                spec["date"],
                np.nan
            )


            if (
                pd.notna(home_date)
                and
                pd.notna(pcr_date)
                and
                home_date > pcr_date
            ):
                continue


            pos = row.get(
                spec["positive"],
                np.nan
            )

            neg = row.get(
                spec["negative"],
                np.nan
            )

            inv = row.get(
                spec["invalid"],
                np.nan
            )


            if pos == 1:

                statuses.append(
                    "Positive"
                )

            elif neg == 1:

                statuses.append(
                    "Negative"
                )

            elif inv == 1:

                statuses.append(
                    "Invalid"
                )


        if "Positive" in statuses:
            return "Positive"

        if "Negative" in statuses:
            return "Negative"

        if "Invalid" in statuses:
            return "Invalid"

        return "Unknown"


    df[HOME_TEST_VAR] = df.apply(
        derive_home_test_result,
        axis=1
    )


    # --------------------------------------------------------
    # 6. Model B dataset
    # --------------------------------------------------------

    analysis_df = df[
        df[TARGET].isin([0, 1])
    ].copy()


    analysis_df[TARGET] = (
        analysis_df[TARGET]
        .astype(int)
    )


    primary_features = (
        classification.loc[
            classification[
                "Primary_Model_Use"
            ],
            "Variable"
        ]
        .tolist()
    )


    primary_features = [

        feature

        for feature
        in primary_features

        if feature != TARGET
    ]


    def remove_unusable(
        data,
        columns,
        missing_threshold=0.95
    ):

        keep = []

        for column in columns:

            if column not in data.columns:
                continue


            missing_rate = (
                data[column]
                .isna()
                .mean()
            )


            unique_values = (
                data[column]
                .dropna()
                .nunique()
            )


            if (
                missing_rate
                <= missing_threshold
                and
                unique_values > 1
            ):

                keep.append(column)


        return keep


    primary_features = remove_unusable(
        analysis_df,
        primary_features
    )


    MODEL_B_FEATURES = (
        primary_features
        +
        [HOME_TEST_VAR]
    )


    # --------------------------------------------------------
    # 7. LASSO preprocessing
    # --------------------------------------------------------

    def infer_feature_types(
        data,
        features
    ):

        categorical = []
        numeric = []


        for column in features:

            if column == HOME_TEST_VAR:

                categorical.append(column)

                continue


            dtype = (
                datatype_of(column)
                .lower()
            )


            if (
                "categor" in dtype
                or
                "ordinal" in dtype
            ):

                categorical.append(
                    column
                )


            elif (
                "integer" in dtype
                or
                "continuous" in dtype
                or
                "numeric" in dtype
            ):

                numeric.append(
                    column
                )


            elif (
                data[column]
                .dropna()
                .nunique()
                <= 10
            ):

                categorical.append(
                    column
                )


            else:

                numeric.append(
                    column
                )


        return categorical, numeric


    def make_preprocessor(
        data,
        features
    ):

        categorical, numeric = (
            infer_feature_types(
                data,
                features
            )
        )


        cat_pipe = Pipeline([

            (
                "imputer",

                SimpleImputer(
                    strategy="most_frequent"
                )
            ),

            (
                "onehot",

                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ])


        num_pipe = Pipeline([

            (
                "imputer",

                SimpleImputer(
                    strategy="median"
                )
            ),

            (
                "scaler",

                StandardScaler()
            )
        ])


        return ColumnTransformer([

            (
                "cat",
                cat_pipe,
                categorical
            ),

            (
                "num",
                num_pipe,
                numeric
            )
        ])


    # --------------------------------------------------------
    # 8. Fit LASSO Model B
    # --------------------------------------------------------

    X = analysis_df[
        MODEL_B_FEATURES
    ].copy()

    y = analysis_df[
        TARGET
    ].astype(int)


    prep = make_preprocessor(
        analysis_df,
        MODEL_B_FEATURES
    )


    lasso_model = Pipeline([

        (
            "prep",
            clone(prep)
        ),

        (
            "model",

            LogisticRegressionCV(

                Cs=10,

                cv=5,

                penalty="l1",

                solver="saga",

                scoring="roc_auc",

                class_weight="balanced",

                max_iter=10000,

                n_jobs=-1,

                random_state=RANDOM_STATE
            )
        )
    ])


    lasso_model.fit(
        X,
        y
    )


    prep_fitted = (
        lasso_model
        .named_steps["prep"]
    )


    fitted_lasso = (
        lasso_model
        .named_steps["model"]
    )


    encoded_names = (
        prep_fitted
        .get_feature_names_out()
    )


    coefficients = (
        fitted_lasso
        .coef_[0]
    )


    encoded = pd.DataFrame({

        "Encoded_Feature":
            encoded_names,

        "Coefficient":
            coefficients,

        "Abs_Coefficient":
            np.abs(coefficients)
    })


    originals = []


    for encoded_name in encoded_names:

        if encoded_name.startswith(
            "num__"
        ):

            originals.append(
                encoded_name.replace(
                    "num__",
                    "",
                    1
                )
            )

            continue


        core = encoded_name.replace(
            "cat__",
            "",
            1
        )


        match = None


        for column in sorted(
            MODEL_B_FEATURES,
            key=len,
            reverse=True
        ):

            if (
                core == column
                or
                core.startswith(
                    column + "_"
                )
            ):

                match = column

                break


        originals.append(match)


    encoded["Variable"] = originals


    encoded = encoded[
        encoded["Variable"].notna()
    ].copy()


    importance_B = (

        encoded
        .groupby(
            "Variable",
            as_index=False
        )
        .agg(

            LASSO_Importance=(
                "Abs_Coefficient",
                "sum"
            )
        )
        .sort_values(
            "LASSO_Importance",
            ascending=False
        )
    )


    # --------------------------------------------------------
    # 9. Select top 10 Model B predictors
    # --------------------------------------------------------

    TOP_N_BN = 10


    bn_candidates = importance_B[

        (
            importance_B[
                "LASSO_Importance"
            ] > 0
        )

        &

        (
            importance_B[
                "Variable"
            ].isin(
                MODEL_B_FEATURES
            )
        )

    ].copy()


    BN_FEATURES = (

        bn_candidates
        .head(TOP_N_BN)[
            "Variable"
        ]
        .tolist()
    )


    # --------------------------------------------------------
    # 10. Discretize Bayesian variables
    # --------------------------------------------------------

    def discretize_for_bn(
        series,
        variable
    ):

        s = series.copy()

        dtype = (
            datatype_of(variable)
            .lower()
        )

        nonmissing = (
            s.dropna()
        )


        if (
            "categor" in dtype
            or
            "ordinal" in dtype
            or
            nonmissing.nunique() <= 8
        ):

            return (
                s
                .fillna("Unknown")
                .astype(str)
            )


        numeric = pd.to_numeric(
            s,
            errors="coerce"
        )


        if (
            numeric
            .dropna()
            .nunique()
            >= 3
        ):

            try:

                binned = pd.qcut(

                    numeric,

                    q=3,

                    labels=[
                        "Low",
                        "Medium",
                        "High"
                    ],

                    duplicates="drop"
                )


                return (
                    binned
                    .astype(object)
                    .where(
                        binned.notna(),
                        "Unknown"
                    )
                    .astype(str)
                )


            except Exception:

                pass


        return (

            numeric
            .fillna(
                numeric.median()
            )
            .round(0)
            .astype(str)
        )


    bn_data = pd.DataFrame(
        index=analysis_df.index
    )


    for variable in BN_FEATURES:

        bn_data[variable] = (
            discretize_for_bn(
                analysis_df[variable],
                variable
            )
        )


    bn_data[TARGET] = (

        analysis_df[TARGET]
        .astype(int)
        .astype(str)
    )


    # --------------------------------------------------------
    # 11. DEMI temporal relationships
    # --------------------------------------------------------

    def demi_candidate_edges(
        selected_features,
        kb_df
    ):

        selected = set(
            selected_features
        )


        temp = kb_df[

            kb_df[
                "concept_code"
            ].isin(selected)

            &

            kb_df[
                "target_concept_code"
            ].isin(selected)

            &

            (
                kb_df[
                    "concept_code"
                ]
                !=
                kb_df[
                    "target_concept_code"
                ]
            )

        ].copy()


        if temp.empty:

            return pd.DataFrame(

                columns=[

                    "source",
                    "target",
                    "temporal_strength",
                    "association_strength"
                ]
            )


        n11 = (
            temp["n_code_target"]
            .astype(float)
        )

        n10 = (
            temp["n_code_no_target"]
            .astype(float)
        )

        n01 = (
            temp["n_target_no_code"]
            .astype(float)
        )

        n00 = (
            temp["n_no_code_no_target"]
            .astype(float)
        )


        numerator = (
            n11 * n00
            -
            n10 * n01
        )


        denominator = np.sqrt(

            (n11 + n10)

            *

            (n01 + n00)

            *

            (n11 + n01)

            *

            (n10 + n00)
        )


        temp["phi"] = np.where(

            denominator > 0,

            numerator / denominator,

            0
        )


        cb = (

            temp[
                "n_code_before_target"
            ]
            .fillna(0)
            .astype(float)
        )


        tb = (

            temp[
                "n_target_before_code"
            ]
            .fillna(0)
            .astype(float)
        )


        temp["source"] = np.where(

            cb >= tb,

            temp[
                "concept_code"
            ],

            temp[
                "target_concept_code"
            ]
        )


        temp["target"] = np.where(

            cb >= tb,

            temp[
                "target_concept_code"
            ],

            temp[
                "concept_code"
            ]
        )


        temp[
            "temporal_strength"
        ] = np.maximum(
            cb,
            tb
        )


        temp[
            "association_strength"
        ] = (
            temp["phi"]
            .abs()
        )


        return (

            temp
            .sort_values(

                [
                    "temporal_strength",
                    "association_strength"
                ],

                ascending=False
            )

            .drop_duplicates(
                [
                    "source",
                    "target"
                ]
            )

            [
                [
                    "source",
                    "target",
                    "temporal_strength",
                    "association_strength"
                ]
            ]
        )


    demi_edges = demi_candidate_edges(
        BN_FEATURES,
        kb
    )


    # --------------------------------------------------------
    # 12. Build DAG
    # --------------------------------------------------------

    MAX_PREDICTOR_PARENTS = 2


    G = nx.DiGraph()


    G.add_nodes_from(
        BN_FEATURES
        +
        [TARGET]
    )


    for _, row in (
        demi_edges.iterrows()
    ):

        source = row["source"]
        target = row["target"]


        if source == target:
            continue


        if (
            G.in_degree(target)
            >=
            MAX_PREDICTOR_PARENTS
        ):

            continue


        G.add_edge(
            source,
            target
        )


        if not (
            nx.is_directed_acyclic_graph(
                G
            )
        ):

            G.remove_edge(
                source,
                target
            )


    # Limit the number of direct PCR parents to reduce
    # sparse combinations in the Bayesian probability table

    MAX_PCR_PARENTS = 4

    PCR_PARENTS = (
        importance_B[
            importance_B["Variable"].isin(BN_FEATURES)
        ]
        .sort_values(
            "LASSO_Importance",
            ascending=False
        )
        .head(MAX_PCR_PARENTS)["Variable"]
        .tolist()
    )

    for variable in PCR_PARENTS:
        G.add_edge(
            variable,
            TARGET
        )

    # --------------------------------------------------------
    # 13. Fit Bayesian CPDs
    # --------------------------------------------------------

    bn_model = (
        DiscreteBayesianNetwork(
            list(G.edges())
        )
    )


    for node in G.nodes():

        if (
            node
            not in
            bn_model.nodes()
        ):

            bn_model.add_node(
                node
            )


    estimator = BayesianEstimator(

        bn_model,

        bn_data[
            BN_FEATURES
            +
            [TARGET]
        ]
    )


    cpds = estimator.get_parameters(

        prior_type="BDeu",

        equivalent_sample_size=10
    )


    bn_model.add_cpds(
        *cpds
    )


    inference = (
        VariableElimination(
            bn_model
        )
    )


    # --------------------------------------------------------
    # 14. Allowed states
    # --------------------------------------------------------

    allowed_states = {

        variable:
            sorted(
                bn_data[
                    variable
                ]
                .astype(str)
                .unique()
                .tolist()
            )

        for variable
        in BN_FEATURES
    }


    # --------------------------------------------------------
    # Return everything Streamlit needs
    # --------------------------------------------------------

    return {

        "bn_model":
            bn_model,

        "inference":
            inference,

        "bn_data":
            bn_data,

        "BN_FEATURES":
            BN_FEATURES,

        "allowed_states":
            allowed_states,

        "display_name":
            display_name,

        "importance_B":
            importance_B,

        "G":
            G
    }


# ============================================================
# BUILD MODEL
# ============================================================

with st.spinner("Loading COVID-19 diagnostic model..."):
    model_objects = build_covid_model()

bn_model = model_objects["bn_model"]
inference = model_objects["inference"]
bn_data = model_objects["bn_data"]
BN_FEATURES = model_objects["BN_FEATURES"]
allowed_states = model_objects["allowed_states"]
display_name = model_objects["display_name"]
importance_B = model_objects["importance_B"]
G = model_objects["G"]

# DEBUG: Check Bayesian network

st.write("### Model Debug Information")

st.write("Bayesian network valid:", bn_model.check_model())

st.write("BN Features:")
st.write(BN_FEATURES)

st.write("Direct parents of PCR:")
st.write(list(bn_model.get_parents(TARGET)))

debug_result = inference.query(
    variables=[TARGET],
    show_progress=False
)

st.write("Baseline PCR distribution:")
st.write(debug_result)

st.write(
    "PCR target states:",
    debug_result.state_names[TARGET]
)
# ============================================================
# COVID PROBABILITY FUNCTION
# ============================================================

def covid_probability(evidence):

    clean = {}


    for variable, state in (
        evidence.items()
    ):

        if (
            variable
            not in
            BN_FEATURES
        ):

            continue


        state = str(state)


        if (
            state
            not in
            allowed_states[variable]
        ):

            continue


        clean[variable] = state


    result = inference.query(

        variables=[TARGET],

        evidence=clean,

        show_progress=False
    )


    states = [

        str(x)

        for x in
        result.state_names[TARGET]
    ]


    prob_map = dict(

        zip(
            states,
            result.values
        )
    )


    return prob_map.get(

        "1",

        prob_map.get(
            "1.0",
            np.nan
        )
    )


# ============================================================
# USER INTERFACE
# ============================================================

st.title(
    "COVID-19 Home Diagnostic AI"
)

st.write(
    """
    This application estimates the probability of a
    **PCR-positive COVID-19 result** using information
    available at home before a clinic or emergency-room visit.
    """
)


st.info(
    """
    This application is an academic predictive prototype.
    It is not intended to replace professional medical
    diagnosis, clinical evaluation, or laboratory testing.
    """
)


# ============================================================
# DISPLAY MODEL INFORMATION
# ============================================================

with st.expander(
    "About the model"
):

    st.write(
        """
        The Bayesian network uses the top 10 predictors
        identified by the Model B LASSO analysis
        (home information + at-home COVID test).
        """
    )

    st.write(
        "**Selected Bayesian-network predictors:**"
    )

    for variable in BN_FEATURES:

        st.write(
            f"- {display_name(variable)}"
        )


# ============================================================
# PATIENT INPUT
# ============================================================

st.subheader(
    "Enter Home Information"
)


st.write(
    """
    You may leave a variable as **Not provided**.
    Bayesian inference will then estimate the PCR probability
    using the remaining information.
    """
)


evidence = {}


for variable in BN_FEATURES:

    label = display_name(
        variable
    )


    states = allowed_states[
        variable
    ]


    # --------------------------------------------
    # Special handling for at-home test
    # --------------------------------------------

    if variable == HOME_TEST_VAR:

        available_test_states = [

            state

            for state in
            [
                "Negative",
                "Positive",
                "Invalid",
                "Unknown"
            ]

            if state in states
        ]


        options = (
            ["No home test"]
            +
            available_test_states
        )


        selection = st.selectbox(

            label,

            options,

            key=variable
        )


        if (
            selection
            !=
            "No home test"
        ):

            evidence[
                variable
            ] = selection


    # --------------------------------------------
    # All other predictors
    # --------------------------------------------

    else:

        options = (
            ["Not provided"]
            +
            states
        )


        selection = st.selectbox(

            label,

            options,

            key=variable
        )


        if (
            selection
            !=
            "Not provided"
        ):

            evidence[
                variable
            ] = selection


# ============================================================
# CALCULATE PROBABILITY
# ============================================================

st.divider()


if st.button(
    "Estimate PCR-Positive Probability",
    type="primary"
):

    if len(evidence) == 0:

        st.warning(
            "Please enter at least one piece of home information."
        )

    else:
         # DEBUG: Check whether this exact evidence pattern
        # exists in the training data

        matching = bn_data.copy()

        for variable, state in evidence.items():
            matching = matching[
                matching[variable].astype(str) == str(state)
            ]

        st.write("Exact matching training records:", len(matching))

        if len(matching) > 0:
            st.write(
                "PCR-positive rate among exact matches:",
                round(
                    (matching[TARGET].astype(str) == "1").mean(),
                    3
                )
            )
        else:
            st.warning(
                "This exact combination of predictor values was not "
                "observed in the training dataset."
            )

        probability = covid_probability(
            evidence
        )


        st.subheader(
            "Estimated Result"
        )


        st.metric(

            label=(
                "Estimated probability "
                "of PCR-positive COVID-19"
            ),

            value=(
                f"{probability * 100:.1f}%"
            )
        )


        # ----------------------------------------
        # Compare with and without home test
        # ----------------------------------------

        if (
            HOME_TEST_VAR
            in evidence
        ):

            evidence_without_test = (
                evidence.copy()
            )


            evidence_without_test.pop(
                HOME_TEST_VAR
            )


            probability_without_test = (
                covid_probability(
                    evidence_without_test
                )
            )


            st.write(
                "**Comparison with and without the home test:**"
            )


            comparison = pd.DataFrame({

                "Scenario": [

                    "Home information only",

                    (
                        "Home information "
                        "+ at-home test"
                    )
                ],

                "Estimated PCR-positive probability": [

                    f"{probability_without_test * 100:.1f}%",

                    f"{probability * 100:.1f}%"
                ]
            })


            st.dataframe(
                comparison,
                hide_index=True,
                use_container_width=True
            )


        # ----------------------------------------
        # Show evidence entered
        # ----------------------------------------

        with st.expander(
            "Evidence used in this estimate"
        ):

            evidence_table = (
                pd.DataFrame(
                    [
                        {
                            "Predictor":
                                display_name(
                                    variable
                                ),

                            "Entered state":
                                state
                        }

                        for (
                            variable,
                            state
                        )
                        in evidence.items()
                    ]
                )
            )


            st.dataframe(
                evidence_table,
                hide_index=True,
                use_container_width=True
            )


# ============================================================
# SHOW TOP MODEL B FEATURES
# ============================================================

st.divider()

st.subheader(
    "Top Model B LASSO Predictors"
)


top_features = (
    importance_B
    .head(10)
    .copy()
)


top_features[
    "Predictor"
] = (
    top_features[
        "Variable"
    ]
    .apply(
        display_name
    )
)


top_features = top_features[

    [
        "Predictor",
        "LASSO_Importance"
    ]
]


st.dataframe(
    top_features,
    hide_index=True,
    use_container_width=True
)


# ============================================================
# FOOTER
# ============================================================

st.caption(
    """
    COVID-19 Home Diagnostic AI |
    Python • scikit-learn • NetworkX • pgmpy • Streamlit
    """
)
