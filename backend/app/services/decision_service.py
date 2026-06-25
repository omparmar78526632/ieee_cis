class DecisionService:

    @staticmethod
    def generate_recommendation(data):

        recommendations = []

        # Efficiency
        if data.predicted_power >= 700:
            efficiency = "Excellent"

        elif data.predicted_power >= 500:
            efficiency = "Good"

        elif data.predicted_power >= 300:
            efficiency = "Average"

        else:
            efficiency = "Poor"

        # Rules

        if data.humidity > 80:
            recommendations.append(
                "High humidity detected. Panel efficiency may decrease."
            )

        if data.temperature > 40:
            recommendations.append(
                "High temperature detected. Monitor panel temperature."
            )

        if data.irradiation < 500:
            recommendations.append(
                "Low solar irradiation. Reduced generation expected."
            )

        if data.wind_speed > 10:
            recommendations.append(
                "High wind speed. Inspect mounting structure."
            )

        if data.predicted_power < 400:
            recommendations.append(
                "Predicted generation is below expected."
            )

        if len(recommendations) == 0:
            recommendations.append(
                "Weather conditions are favorable."
            )

        return {

            "predicted_power": data.predicted_power,

            "efficiency": efficiency,

            "recommendations": recommendations

        }