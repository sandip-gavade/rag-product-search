from django.contrib import admin

from .models import Product


@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ('external_id', 'title', 'category', 'price', 'updated_at')
    list_filter = ('category',)
    search_fields = ('title', 'description', 'external_id')
