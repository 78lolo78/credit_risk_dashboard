# ------------------------------------
# import packages
# ------------------------------------
import requests
import json
from pandas import json_normalize
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import seaborn as sns
import shap
from shap.plots import waterfall
import matplotlib.pyplot as plt
from PIL import Image
# ----------------------------------------------------
# main function
# ----------------------------------------------------
def main():
    # ------------------------------------------------
    # local API (à remplacer par l'adresse de l'application déployée)
    # -----------------------------------------------
    API_URL = "https://loan-flask-api.herokuapp.com/app/"
    # Local URL: http://localhost:8501
    # -----------------------------------------------
    # Configuration of the streamlit page
    # -----------------------------------------------
    st.set_page_config(page_title='Prédiction de défaut de remboursement de prêt',
                       page_icon='🧊',
                       layout='centered',
                       initial_sidebar_state='auto')
    # Display the title
    st.title('Prédiction de défaut de remboursement de prêt')
    st.subheader("Vérifions si le client demandeur d'un prêt est en capacité de remboursement au moment de la demande?💸 "
                 "Cette application de machine learning va vous aider à faire une prédiction pour vous aider dans la prise de décision!")

    # Display the LOGO
    # files = os.listdir('Image_logo')
    # for file in files:
    img = Image.open("LOGO.png")
    st.sidebar.image(img, width=250)

    # # Display the loan image
    # files = os.listdir('Image_loan')
    # for file in files:
    img = Image.open("loan.png")
    st.image(img, width=100)

    # ====================================================================
    # FOOTER
    # ====================================================================
    html_line="""
    <br>
    <br>
    <br>
    <br>
    <hr style= "  display: block;
    margin-top: 0.5em;
    margin-bottom: 0.5em;
    margin-left: auto;
    margin-right: auto;
    border-style: inset;
    border-width: 1.5px;">
    <p style="color:Gray; text-align: right; font-size:12px;">Auteur : Laurent Cagniart - 01/2023</p>
    """
    st.markdown(html_line, unsafe_allow_html=True)

    # Functions
    # ----------
    def get_list_display_features(f, def_n):
        all_feat = f
        n = st.slider("Nb of features to display",
                      min_value=2, max_value=40,
                      value=def_n, step=None, format=None)

        disp_cols = list(get_features_importances().sort_values(ascending=False).iloc[:n].index)

        box_cols = st.multiselect(
            'Choose the features to display:',
            sorted(all_feat),
            default=disp_cols)
        return box_cols

    ###############################################################################
    #                      LIST OF API REQUEST FUNCTIONS
    ###############################################################################
    # Get list of ID (cached)
    @st.cache(suppress_st_warning=True)
    def get_id_list():
        # URL of the sk_id API
        id_api_url = API_URL + "id/"
        # Requesting the API and saving the response
        response = requests.get(id_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of "ID" from the content
        id_customers = pd.Series(content['data']).values
        return id_customers

    # Get selected customer's data (cached)
    # local test api : http://127.0.0.1:5000/app/data_cust/?SK_ID_CURR=27141
    data_type = []

    @st.cache
    def get_selected_cust_data(selected_id):
        # URL of the sk_id API
        data_api_url = API_URL + "data_cust/?SK_ID_CURR=" + str(selected_id)
        # Requesting the API and saving the response
        response = requests.get(data_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        x_custom = pd.DataFrame(content['data'])
        # x_cust = json_normalize(content['data'])
        y_customer = (pd.Series(content['y_cust']).rename('TARGET'))
        # y_customer = json_normalize(content['y_cust'].rename('TARGET'))
        return x_custom, y_customer

    @st.cache
    def get_all_cust_data():
        # URL of the sk_id API
        data_api_url = API_URL + "all_proc_data_tr/"
        # Requesting the API and saving the response
        response = requests.get(data_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))  #
        x_all_cust = json_normalize(content['X_train'])  # Results contain the required data
        y_all_cust = json_normalize(content['y_train'].rename('TARGET'))  # Results contain the required data
        return x_all_cust, y_all_cust

    # Get score (cached)
    @st.cache
    def get_score_model(selected_id):
        # URL of the sk_id API
        score_api_url = API_URL + "scoring_cust/?SK_ID_CURR=" + str(selected_id)
        # Requesting the API and saving the response
        response = requests.get(score_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # Getting the values of "ID" from the content
        score_model = (content['score'])
        threshold = content['thresh']
        return score_model, threshold

    # Get list of shap_values (cached)
    # local test api : http://127.0.0.1:5000/app/shap_val//?SK_ID_CURR=27141
    @st.cache
    def values_shap(selected_id):
        # URL of the sk_id API
        shap_values_api_url = API_URL + "shap_val/?SK_ID_CURR=" + str(selected_id)
        # Requesting the API and saving the response
        response = requests.get(shap_values_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of "ID" from the content
        shapvals = pd.DataFrame(content['shap_val_cust'].values())
        expec_vals = pd.DataFrame(content['expected_vals'].values())
        return shapvals, expec_vals

    #############################################
    #############################################
    # Get list of expected values (cached)
    @st.cache
    def values_expect():
        # URL of the sk_id API
        expected_values_api_url = API_URL + "exp_val/"
        # Requesting the API and saving the response
        response = requests.get(expected_values_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of "ID" from the content
        expect_vals = pd.Series(content['data']).values
        return expect_vals

    # Get list of feature names
    @st.cache
    def feat():
        # URL of the sk_id API
        feat_api_url = API_URL + "feat/"
        # Requesting the API and saving the response
        response = requests.get(feat_api_url)
        # Convert from JSON format to Python dict
        content = json.loads(response.content)
        # Getting the values of "ID" from the content
        features_name = pd.Series(content['data']).values
        return features_name

    # Get the list of feature importances (according to lgbm classification model)
    @st.cache
    def get_features_importances():
        # URL of the aggregations API
        feat_imp_api_url = API_URL + "feat_imp/"
        # Requesting the API and save the response
        response = requests.get(feat_imp_api_url)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert back to pd.Series
        feat_imp = pd.Series(content['data']).sort_values(ascending=False)
        return feat_imp

    # Get data from 20 nearest neighbors in train set (cached)
    @st.cache
    def get_data_neigh(selected_id):
        # URL of the scoring API (ex: SK_ID_CURR = 100005)
        neight_data_api_url = API_URL + "neigh_cust/?SK_ID_CURR=" + str(selected_id)
        # save the response of API request
        response = requests.get(neight_data_api_url)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert data to pd.DataFrame and pd.Series
        data_neig = pd.DataFrame(content['data_neigh'])
        target_neig = (pd.Series(content['y_neigh']).rename('TARGET'))
        return data_neig, target_neig

    # Get data from 1000 nearest neighbors in train set (cached)
    @st.cache
    def get_data_thousand_neigh(selected_id):
        thousand_neight_data_api_url = API_URL + "thousand_neigh/?SK_ID_CURR=" + str(selected_id)
        # save the response of API request
        response = requests.get(thousand_neight_data_api_url)
        # convert from JSON format to Python dict
        content = json.loads(response.content.decode('utf-8'))
        # convert data to pd.DataFrame and pd.Series
        data_thousand_neig = pd.DataFrame(content['X_thousand_neigh'])
        x_custo = pd.DataFrame(content['x_custom'])
        target_thousand_neig = (pd.Series(content['y_thousand_neigh']).rename('TARGET'))
        return data_thousand_neig, target_thousand_neig, x_custo

    #############################################################################
    #                          Selected id
    #############################################################################
    # list of customer's ID's
    cust_id = get_id_list()
    # Selected customer's ID
    selected_id = st.sidebar.selectbox('Selectionner ID client dans la liste:', cust_id, key=18)
    st.write('ID client sélectionné = ', selected_id)

    ############################################################################
    #                           Graphics Functions
    ############################################################################
    # Global SHAP SUMMARY
    @st.cache
    def shap_summary():
        return shap.summary_plot(shap_vals, feature_names=features)

    # Local SHAP Graphs
    @st.cache
    def waterfall_plot(nb, ft, expected_val, shap_val):
        return shap.plots._waterfall.waterfall_legacy(expected_val, shap_val[0, :],
                                                      max_display=nb, feature_names=ft)

    # Local SHAP Graphs
    @st.cache(allow_output_mutation=True)  #
    def force_plot():
        shap.initjs()
        return shap.force_plot(expected_vals[0][0], shap_vals[0, :], matplotlib=True)

    # Gauge Chart
    @st.cache
    def gauge_plot(scor, th):
        scor = int(scor * 100)
        th = int(th * 100)

        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=scor,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Probabilité de défaut du client", 'font': {'size': 25}},
            delta={'reference': int(th), 'increasing': {'color': 'Crimson'}, 'decreasing': {'color': 'Green'}},
            gauge={
                'axis': {'range': [None, int(100)], 'tickwidth': 3, 'tickcolor': 'darkblue'},
                'bar': {'color': 'white', 'thickness': 0.25},
                'bgcolor': 'white',
                'borderwidth': 2,
                'bordercolor': 'gray',
                'steps': [
                    {'range': [0, 20], 'color': 'LimeGreen'},
                    {'range': [20, int(th) - 5], 'color': 'Green'},
                    {'range': [int(th) - 5, int(th) +5], 'color': 'Orange'},
                    {'range': [int(th) + 5, 75], 'color': 'red'},
                    {'range': [75, 100], 'color': 'Crimson'}],
                'threshold': {
                    'line': {'color': 'white', 'width': 10},
                    'thickness': 0.8,
                    'value': int(scor)}}))

        fig.update_layout(paper_bgcolor='white', font={'color': "darkblue", 'family': "Arial"}, margin=dict(l=0, r=0, b=0, t=0, pad=0))
        return fig



    ##############################################################################
    #                         Customer's data checkbox
    ##############################################################################
    if st.sidebar.checkbox("Données Client"):
        st.markdown('Données du client sélectionné :')
        data_selected_cust, y_cust = get_selected_cust_data(selected_id)
        st.write(data_selected_cust)
    ##############################################################################
    #                         Model's decision checkbox
    ##############################################################################
    if st.sidebar.checkbox("Décision du modèle", key=38):
        # Get score & threshold model
        score, threshold_model = get_score_model(selected_id)
        # Display score (default probability)
        st.write('Probabilité de défaut : {:.0f}%'.format(score * 100))
        # Display default threshold
        #st.write('Seuil de probabilité de défaut du modèle : {:.0f}%'.format(threshold_model * 100))  #
        # Compute decision according to the best threshold (False= loan accepted, True=loan refused)
        if score >= threshold_model:
            decision = "Crédit rejeté"
        else:
            decision = "Crédit accordé"
        st.write("Decision :", decision)
        ##########################################################################
        #              Display customer's gauge meter chart (checkbox)
        ##########################################################################
        figure = gauge_plot(score, threshold_model)
        st.write(figure)
        # Add markdown
        st.markdown('_Jauge - probabilité de défaut pour le client sélectionné._')
        st.markdown('_le chiffre à côté du triangle indique l écart par rapport au seuil de classification._')
        expander = st.expander("A propos du modèle de classification...")
        expander.write("La prédiction a été réalisée en utilisant un modèle de classification dit Light Gradient Boosting Model")
        
        ##########################################################################
        #                 Display local SHAP waterfall checkbox
        ##########################################################################
        if st.checkbox('Montrer l interprétation locale de la prédiction par SHAP waterfall plot', key=25):
            with st.spinner('SHAP waterfall plots en cours de construction..... Merci de patienter.......'):
                # Get Shap values for customer & expected values
                shap_vals, expected_vals = values_shap(selected_id)
                # index_cust = customer_ind(selected_id)
                # Get features names
                features = feat()
                # st.write(features)
                nb_features = st.slider("Number of features to display",
                                        min_value=2,
                                        max_value=50,
                                        value=10,
                                        step=None,
                                        format=None,
                                        key=14)
                # draw the waterfall graph (only for the customer with scaling
                waterfall_plot(nb_features, features, expected_vals[0][0], shap_vals.values)

                plt.gcf()
                st.pyplot(plt.gcf())
                # Add markdown
                st.markdown('_SHAP waterfall Plot pour le client sélectionné._')
                # Add details title
                expander = st.expander("Explication sur le SHAP waterfall  plot...")
                # Add explanations
                expander.write("Le SHAP waterfall  plot ci-dessus montre \
                comment est construite la prédiction individuelle du client sélectionné.\
                Le bas du graphique commence par la valeur attendue en sortie du modèle \
                (i.e. la valeur obtenue si aucune information sur les features étaient fournies), et ensuite \
                chaque ligne montre la contribution négative (rouge) ou positive (bleue) \
                (Remarque: plus le score est élevé, plus le risque de défaut est élevé) \
                chaque feature déplace la valeur courante par rapport à la valeur attendue selon le dataset \
                utilisé pour la modélisation et la prédiction.")

        ##########################################################################
        #              Display feature's distribution (Boxplots)
        ##########################################################################
        if st.checkbox('Montrer la distribution des features par classe défaut- non défaut' , key=20):
            st.header('Boxplots des features principales')
            fig, ax = plt.subplots(figsize=(20, 10))
            with st.spinner('Création des Boxplots en cours...merci de patienter.....'):
                # Get Shap values for customer
                shap_vals, expected_vals = values_shap(selected_id)
                # Get features names
                features = feat()
                # Get selected columns
                disp_box_cols = get_list_display_features(features, 2)
                # -----------------------------------------------------------------------------------------------
                # Get targets and data for : all customers + Applicant customer + 20 neighbors of selected customer
                # -----------------------------------------------------------------------------------------------
                # neighbors + Applicant customer :
                data_neigh, target_neigh = get_data_neigh(selected_id)
                data_thousand_neigh, target_thousand_neigh, x_customer = get_data_thousand_neigh(selected_id)

                x_cust, y_cust = get_selected_cust_data(selected_id)
                x_customer.columns = x_customer.columns.str.split('.').str[0]
                # Target imputation (0 : 'repaid (....), 1 : not repaid (....)
                # -------------------------------------------------------------
                target_neigh = target_neigh.replace({0: 'repaid (neighbors)',
                                                     1: 'not repaid (neighbors)'})
                y_cust = y_cust.replace({0: 'repaid (customer)',
                                         1: 'not repaid (customer)'})

                # y_cust.rename(columns={'10006':'TARGET'}, inplace=True)
                
                # ------------------------------
                # Get 1000 neighbors personal data
                # ------------------------------
                df_thousand_neigh = pd.concat([data_thousand_neigh[disp_box_cols], target_thousand_neigh], axis=1)
                df_melt_thousand_neigh = df_thousand_neigh.reset_index()
                df_melt_thousand_neigh.columns = ['index'] + list(df_melt_thousand_neigh.columns)[1:]
                df_melt_thousand_neigh = df_melt_thousand_neigh.melt(id_vars=['index', 'TARGET'],
                                                                     value_vars=disp_box_cols,
                                                                     var_name="variables",  # "variables",
                                                                     value_name="values")

                sns.boxplot(data=df_melt_thousand_neigh, x='variables', y='values',
                            hue='TARGET', linewidth=1, width=0.4,
                            palette=['tab:green', 'tab:red'], showfliers=False,
                            saturation=0.5, ax=ax)

                # ------------------------------
                # Get 20 neighbors personal data
                # ------------------------------
                df_neigh = pd.concat([data_neigh[disp_box_cols], target_neigh], axis=1)
                df_melt_neigh = df_neigh.reset_index()
                df_melt_neigh.columns = ['index'] + list(df_melt_neigh.columns)[1:]
                df_melt_neigh = df_melt_neigh.melt(id_vars=['index', 'TARGET'],
                                                   value_vars=disp_box_cols,
                                                   var_name="variables",  # "variables",
                                                   value_name="values")

                sns.swarmplot(data=df_melt_neigh, x='variables', y='values', hue='TARGET', linewidth=1,
                              palette=['darkgreen', 'darkred'], marker='o', size=15, edgecolor='k', ax=ax)

                # -----------------------
                # Applicant customer data
                # -----------------------
                df_selected_cust = pd.concat([x_customer[disp_box_cols], y_cust], axis=1)
                # st.write("df_sel_cust :", df_sel_cust)
                df_melt_sel_cust = df_selected_cust.reset_index()
                df_melt_sel_cust.columns = ['index'] + list(df_melt_sel_cust.columns)[1:]
                df_melt_sel_cust = df_melt_sel_cust.melt(id_vars=['index', 'TARGET'],
                                                         value_vars=disp_box_cols,
                                                         var_name="variables",
                                                         value_name="values")

                sns.swarmplot(data=df_melt_sel_cust, x='variables', y='values',
                              linewidth=1, color='y', marker='o', size=20,
                              edgecolor='k', label='client sélectionné', ax=ax)

                # legend
                h, _ = ax.get_legend_handles_labels()
                ax.legend(handles=h[:5])

                plt.xticks(rotation=20, ha='right')
                plt.show()

                st.write(fig)  # st.pyplot(fig) # the same

                plt.xticks(rotation=20, ha='right')
                plt.show()

                st.markdown('_Distribution des features principales pour la base client (sample),\
                20 clients similaires et le client sélectionné_')

                expander = st.expander("Explications sur les graphiques de distribution...")
                expander.write("Ces boîtes à moustaches montre la distibution des valeurs des features preprocessées\
                utilisées par le modèle pour faire la prédiction. La boîte verte montre les clients qui ont remboursé leur prêt \
                , et la rouge, ceux qui ont été en dafut sur le remboursmeent de leur prêt. Sur ces boîtes, ont été ajoutés\
                des marqueurs situant précisément les 20 clients similaires (dataset de training) avec le même codage couleur,\
                le marqueur du client sélectionné est de couleur jaune pour le distinguer.")


if __name__ == "__main__":
    main()
