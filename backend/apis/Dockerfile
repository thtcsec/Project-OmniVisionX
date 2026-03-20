# OmniVision API - .NET 9
FROM mcr.microsoft.com/dotnet/aspnet:9.0

WORKDIR /app

COPY ["Omni.API.csproj", "./"]
RUN dotnet restore

COPY . ./
RUN dotnet publish -c Release -o /app/publish

EXPOSE 8080

ENTRYPOINT ["dotnet", "/app/publish/Omni.API.dll"]
