from rest_framework import serializers

from catalog.models import Product


class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = ["id", "external_id", "title", "description", "category", "price", "attributes"]


class SearchResultSerializer(serializers.Serializer):
    """A Product plus its hybrid-search score breakdown.

    The breakdown (rather than just a single fused number) is what makes
    the endpoint debuggable — you can tell whether a result ranked highly
    because of semantic similarity, a keyword match, or both.
    """

    product = ProductSerializer()
    vector_score = serializers.FloatField(allow_null=True)
    keyword_score = serializers.FloatField(allow_null=True)
    fused_score = serializers.FloatField()
