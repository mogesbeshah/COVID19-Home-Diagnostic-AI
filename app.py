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

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegressionCV


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="COVID-19 Home Diagnostic AI",
    page_icon="🩺",
    layout="centered"
)

RANDOM_STATE = 42

TARGET = "PCR Test Positive"

HOME_TEST_VAR = "HOME_TEST_RESULT_PRE_PCR"


# ============================================================
# BUILD PREDICTIVE MODELS
# ============================================================

@st.cache_resource
def build_models():

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

    df.columns = df.columns.str.strip()
    dictionary.columns = dictionary.columns.str.strip()


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
    # 3. Display labels
    # --------------------------------------------------------

    LABEL_OVERRIDES = {

        HOME_TEST_VAR:
            "At-home COVID test",

        "30158-Symtpom_Neuro-8":
            "Loss of taste",

        "30158-Symtpom_Neuro-7":
            "Loss of smell",

        "30166-Prev_exposure-1":
            "Recent COVID exposure",

        "32136-vaccine_didyou":
            "Vaccination status"
    }


    def short_description(var):

        if var in LABEL_OVERRIDES:
            return LABEL_OVERRIDES[var]

        desc = description_of(var)
        prompt = prompt_of(var)

        text = prompt if prompt else desc

        if "?" in text:

            specific = text.split("?")[-1].strip()

            if specific:
                return specific

        if ";" in text:

            specific = text.split(";")[-1].strip()

            if specific:
                return specific

        return desc


    # --------------------------------------------------------
    # 4. Identify home-available variables
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

        status, use = classify_variable(var)

        classification_rows.append({

            "Variable":
                var,

            "Home_Classification":
                status,

            "Primary_Model_Use":
                use
        })


    classification = pd.DataFrame(
        classification_rows
    )


    # --------------------------------------------------------
    # 5. Create home-test variable
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
    # 6. PCR-known analysis population
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

        for feature in primary_features

        if feature != TARGET
    ]


    # --------------------------------------------------------
    # 7. Remove unusable predictors
    # --------------------------------------------------------

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

                keep.append(
                    column
                )


        return keep


    primary_features = remove_unusable(
        analysis_df,
        primary_features
    )


    # --------------------------------------------------------
    # 8. Model A and Model B
    # --------------------------------------------------------

    MODEL_A_FEATURES = (
        primary_features
    )


    MODEL_B_FEATURES = (
        primary_features
        +
        [HOME_TEST_VAR]
    )


    # --------------------------------------------------------
    # 9. Infer feature types
    # --------------------------------------------------------

    def infer_feature_types(
        data,
        features
    ):

        categorical = []

        numeric = []


        for column in features:

            if column == HOME_TEST_VAR:

                categorical.append(
                    column
                )

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


    # --------------------------------------------------------
    # 10. Preprocessor
    # --------------------------------------------------------

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
    # 11. Fit LASSO function
    # --------------------------------------------------------

    def fit_lasso(features):

        X = analysis_df[
            features
        ].copy()


        y = analysis_df[
            TARGET
        ].astype(int)


        prep = make_preprocessor(
            analysis_df,
            features
        )


        model = Pipeline([

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


        model.fit(
            X,
            y
        )


        return model


    # --------------------------------------------------------
    # 12. Fit Model A and Model B
    # --------------------------------------------------------

    lasso_A = fit_lasso(
        MODEL_A_FEATURES
    )


    lasso_B = fit_lasso(
        MODEL_B_FEATURES
    )


    # --------------------------------------------------------
    # 13. Model B LASSO importance
    # --------------------------------------------------------

    prep_B = (
        lasso_B
        .named_steps["prep"]
    )


    fitted_B = (
        lasso_B
        .named_steps["model"]
    )


    encoded_names = (
        prep_B
        .get_feature_names_out()
    )


    coefficients = (
        fitted_B
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


        originals.append(
            match
        )


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
    # 14. Five interpretable Streamlit inputs
    # --------------------------------------------------------

    APP_FEATURES = [

        HOME_TEST_VAR,

        "30158-Symtpom_Neuro-8",   # Loss of taste

        "30158-Symtpom_Neuro-7",   # Loss of smell

        "30166-Prev_exposure-1",    # Recent exposure

        "32136-vaccine_didyou"      # Vaccination
    ]


    APP_FEATURES = [

        variable

        for variable in APP_FEATURES

        if (
            variable in analysis_df.columns
            or
            variable == HOME_TEST_VAR
        )
    ]


    # --------------------------------------------------------
    # 15. Observed states for Streamlit inputs
    # --------------------------------------------------------

    observed_states = {}


    for variable in APP_FEATURES:

        values = (

            analysis_df[
                variable
            ]

            .dropna()

            .unique()

            .tolist()
        )


        try:

            values = sorted(
                values
            )

        except Exception:

            values = sorted(
                values,
                key=lambda x: str(x)
            )


        observed_states[
            variable
        ] = values


    # --------------------------------------------------------
    # Return app objects
    # --------------------------------------------------------

    return {

        "analysis_df":
            analysis_df,

        "lasso_A":
            lasso_A,

        "lasso_B":
            lasso_B,

        "MODEL_A_FEATURES":
            MODEL_A_FEATURES,

        "MODEL_B_FEATURES":
            MODEL_B_FEATURES,

        "APP_FEATURES":
            APP_FEATURES,

        "observed_states":
            observed_states,

        "importance_B":
            importance_B,

        "display_name":
            short_description
    }


# ============================================================
# LOAD MODELS
# ============================================================

with st.spinner(
    "Loading COVID-19 predictive models..."
):

    model_objects = build_models()


analysis_df = (
    model_objects[
        "analysis_df"
    ]
)

lasso_A = (
    model_objects[
        "lasso_A"
    ]
)

lasso_B = (
    model_objects[
        "lasso_B"
    ]
)

MODEL_A_FEATURES = (
    model_objects[
        "MODEL_A_FEATURES"
    ]
)

MODEL_B_FEATURES = (
    model_objects[
        "MODEL_B_FEATURES"
    ]
)

APP_FEATURES = (
    model_objects[
        "APP_FEATURES"
    ]
)

observed_states = (
    model_objects[
        "observed_states"
    ]
)

importance_B = (
    model_objects[
        "importance_B"
    ]
)

display_name = (
    model_objects[
        "display_name"
    ]
)


# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def create_patient_row(model_features, evidence):

    # Create one empty patient row that can hold
    # both numeric values and text/categorical values
    patient = pd.DataFrame(
        {
            variable: pd.Series([np.nan], dtype="object")
            for variable in model_features
        }
    )

    # Add the information entered by the user
    for variable, value in evidence.items():
        if variable in patient.columns:
            patient.at[0, variable] = value

    return patient


def predict_model_A(
    evidence
):

    # Remove home test because
    # Model A must not use it

    home_evidence = {

        variable:
            value

        for variable, value
        in evidence.items()

        if variable != HOME_TEST_VAR
    }


    patient = create_patient_row(

        MODEL_A_FEATURES,

        home_evidence

    )


    probability = (
        lasso_A
        .predict_proba(
            patient
        )[0, 1]
    )


    return float(
        probability
    )


def predict_model_B(
    evidence
):

    patient = create_patient_row(

        MODEL_B_FEATURES,

        evidence

    )


    probability = (
        lasso_B
        .predict_proba(
            patient
        )[0, 1]
    )


    return float(
        probability
    )


# ============================================================
# FRIENDLY INPUT LABELS
# ============================================================

def friendly_state(value, variable=None):

    # Special labels for vaccination status
    if variable == "32136-vaccine_didyou":

        vaccine_labels = {
            1: "Yes",
            2: "No",
            4: "Not sure — participated in a COVID-19 vaccination trial",
            999: "Prefer not to answer"
        }

        try:
            numeric_value = int(float(value))

            if numeric_value in vaccine_labels:
                return vaccine_labels[numeric_value]

        except (ValueError, TypeError):
            pass

    # General formatting for all other variables
    if value is None:
        return "Not provided"

    if isinstance(
        value,
        (int, float, np.integer, np.floating)
    ):

        if pd.isna(value):
            return "Not provided"

        if float(value) == 0:
            return "No"

        if float(value) == 1:
            return "Yes"

        if float(value).is_integer():
            return str(int(value))

    return str(value)
# ============================================================
# PAGE CONTENT
# ============================================================

st.title(
    "COVID-19 Home Diagnostic AI"
)


st.write(
    """
    This academic AI prototype estimates the probability of a
    **PCR-positive COVID-19 result** using information available
    at home before a clinic or emergency-room visit.
    """
)


st.info(
    """
    This application is intended for academic and research
    demonstration only. It should not be used as a substitute
    for professional medical evaluation or diagnostic testing.
    """
)


# ============================================================
# ABOUT MODEL
# ============================================================

with st.expander(
    "About the predictive models"
):

    st.markdown(
        """
        **Model A — Home Information Only**

        Uses eligible information available before the
        home-test result.

        **Model B — Home Information + At-Home Test**

        Uses the same home information plus the derived
        pre-PCR at-home COVID test result.

        Both probabilities shown in this application are
        generated using LASSO logistic regression models.
        """
    )


# ============================================================
# PATIENT INPUT
# ============================================================

st.subheader(
    "Enter Home Information"
)


st.write(
    """
    Enter the information available for the patient.
    Fields may be left as **Not provided**.
    """
)


evidence = {}


for variable in APP_FEATURES:

    label = display_name(
        variable
    )


    states = observed_states[
        variable
    ]


    # --------------------------------------------------------
    # Home test
    # --------------------------------------------------------

    if variable == HOME_TEST_VAR:

        test_options = [
            None
        ]


        preferred_order = [

            "Negative",

            "Positive",

            "Invalid",

            "Unknown"
        ]


        for state in preferred_order:

            if state in states:

                test_options.append(
                    state
                )


        selection = st.selectbox(

            label,

            test_options,

            format_func=lambda x:
                (
                    "No home test"
                    if x is None
                    else str(x)
                ),

            key=variable
        )


        if selection is not None:

            evidence[
                variable
            ] = selection


    # --------------------------------------------------------
    # Other home information
    # --------------------------------------------------------

    else:

        options = [
            None
        ] + states


        selection = st.selectbox(

            label,

            options,

            format_func=lambda x, v=variable: friendly_state(x, v),

            key=variable
        )


        if selection is not None:

            evidence[
                variable
            ] = selection


# ============================================================
# ESTIMATE PROBABILITY
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

        # ----------------------------------------------------
        # Model A
        # ----------------------------------------------------

        probability_A = predict_model_A(
            evidence
        )


        # ----------------------------------------------------
        # Model B only when home-test information exists
        # ----------------------------------------------------

        has_home_test = (
            HOME_TEST_VAR
            in evidence
        )


        if has_home_test:

            probability_B = predict_model_B(
                evidence
            )

        else:

            probability_B = None


        # ----------------------------------------------------
        # Primary result
        # ----------------------------------------------------

        st.subheader(
            "Estimated Result"
        )


        if has_home_test:

            st.metric(

                label=(
                    "Estimated probability of "
                    "PCR-positive COVID-19 "
                    "(Model B)"
                ),

                value=(
                    f"{probability_B * 100:.1f}%"
                )
            )


        else:

            st.metric(

                label=(
                    "Estimated probability of "
                    "PCR-positive COVID-19 "
                    "(Model A)"
                ),

                value=(
                    f"{probability_A * 100:.1f}%"
                )
            )


        # ----------------------------------------------------
        # Comparison
        # ----------------------------------------------------

        if has_home_test:

            st.write(
                "**Comparison with and without the home test:**"
            )


            comparison = pd.DataFrame({

                "Scenario": [

                    "Model A — Home information only",

                    (
                        "Model B — Home information "
                        "+ at-home test"
                    )
                ],

                "Estimated PCR-positive probability": [

                    f"{probability_A * 100:.1f}%",

                    f"{probability_B * 100:.1f}%"
                ]
            })


            st.dataframe(

                comparison,

                hide_index=True,

                use_container_width=True
            )


            difference = (
                probability_B
                -
                probability_A
            )


            st.write(

                "Change after adding the at-home test: "
                f"**{difference * 100:+.1f} percentage points**"
            )

            if difference > 0:

                st.write(
                    "The at-home test increased the estimated "
                    "probability of a PCR-positive result."
            )

            elif difference < 0:

                st.write(
                "The at-home test decreased the estimated "
                "probability of a PCR-positive result."
            )

            else:

                st.write(
                    "The at-home test did not change the estimated "
                    "probability of a PCR-positive result."
            )

        # ----------------------------------------------------
        # Evidence table
        # ----------------------------------------------------

        with st.expander(
            "Information used in this estimate"
        ):

            evidence_table = pd.DataFrame(

                [

                    {

                        "Predictor":
                            display_name(
                                variable
                            ),

                        "Entered value":
                            friendly_state(
                                value,
                                variable
                            )

                    }

                    for variable, value
                    in evidence.items()

                ]

            )


            st.dataframe(

                evidence_table,

                hide_index=True,

                use_container_width=True
            )


# ============================================================
# MODEL B FEATURE IMPORTANCE
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
# PROJECT NOTE
# ============================================================

with st.expander(
    "Bayesian-network component"
):

    st.write(
        """
        The full project notebook also constructs a Bayesian
        network using LASSO-selected variables, DEMI temporal
        relationships, NetworkX, and pgmpy.

        The Bayesian network is retained as the probabilistic
        network-analysis component of the project. The interactive
        Streamlit prediction tool uses the fitted LASSO models
        because they can generate estimates for new combinations
        of patient information without requiring an exact
        Bayesian CPT combination to have been observed in the
        training data.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.caption(
    """
    COVID-19 Home Diagnostic AI |
    Python • scikit-learn • LASSO • Streamlit
    """
)
