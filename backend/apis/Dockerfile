# syntax=docker/dockerfile:1.4
# Layer order: restore when .csproj changes only. NuGet cache = faster rebuilds (DOCKER_BUILDKIT=1).
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

COPY ["Omni.API.csproj", "./"]
RUN --mount=type=cache,target=/root/.nuget/packages \
    dotnet restore "./Omni.API.csproj"

COPY . ./
# Do not use --no-restore here: with BuildKit NuGet cache mounts, publish can fail
# (NETSDK1064: analyzers / packages "not found") across separate RUN layers.
RUN --mount=type=cache,target=/root/.nuget/packages \
    dotnet restore "./Omni.API.csproj" \
    && dotnet publish "./Omni.API.csproj" -c Release -o /app/publish --no-restore

FROM mcr.microsoft.com/dotnet/aspnet:9.0 AS final
WORKDIR /app
COPY --from=build /app/publish .
EXPOSE 8080
ENV ASPNETCORE_URLS=http://+:8080
ENTRYPOINT ["dotnet", "Omni.API.dll"]
